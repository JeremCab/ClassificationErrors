import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from tqdm import tqdm

import copy

import numpy as np
from scipy.sparse import vstack, csr_matrix
from scipy.optimize import OptimizeResult, Bounds, linprog, minimize, check_grad, SR1, BFGS
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import csr_matrix, vstack, lil_matrix

import cyipopt
import nlopt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from utils.dataset import create_dataset
from utils.network import load_network, SmallConvNet, SmallDenseNet
from preprocessing import LossHead, create_comparing_network, eval_one_sample
from preprocessing import eval_one_sample, squeeze_network, prune_network, get_subnetwork, truncate_after_last_relu

TOL = 1e-8   # almost 0: to check the bounds
TOL2 = 1e-9  # almost 0: to check the bounds

def create_c(comp_net, inputs):
    assert inputs.shape[0] == 1 # one sample in a batch

    # wide_inputs = torch.hstack([inputs, inputs]) TODO: delete this line

    # reduce and squeeze comp_net 
    saturations = eval_one_sample(comp_net, inputs)
    target_net = squeeze_network(prune_network(comp_net, saturations))

    W = target_net[-1].weight.data
    b = target_net[-1].bias.data

    assert W.shape[0] == 1
    
    c = torch.hstack([b, W.flatten()])

    return c

def create_upper_bounds(net, inputs):

    # extract the sequential 
    #    net = next(iter(net.children())) NO NEED FOR COMPNET
    assert isinstance(net, nn.Sequential)
    
    saturations = eval_one_sample(net, inputs)

    A_list = []
    bound_list = [] 
    # Compute the shortcut weights of each layer j (squeeze the subnet up to layer j) => W_ji and W_j0
    for i, saturation in enumerate(saturations):
        subnet = get_subnetwork(net, i)
        if i == 0:
            target = subnet
        else:
            target = squeeze_network(prune_network(subnet, saturations[:i])) # network prunning could be done once for all before...

        W = target[-1].weight.data
        b = target[-1].bias.data

        # saturation: True ~ U, False ~ S   
        W_lower = W[torch.logical_not(saturation).flatten()]
        b_lower = b[torch.logical_not(saturation).flatten()].reshape(-1, 1)
        W_higher = W[saturation.flatten()]
        b_higher = b[saturation.flatten()].reshape(-1, 1)

        bound_for_lower = torch.full((W_lower.shape[0],), -TOL, dtype=torch.float64)
        bound_for_higher = torch.full((W_higher.shape[0],), -TOL, dtype=torch.float64)
        
        W = torch.vstack([W_lower, -1*W_higher]) # multiplied by -1 to transform lower bounds into upper bounds (linprog pkg)
        b = torch.vstack([b_lower, -1*b_higher]) # multiplied by -1 to transform lower bounds into upper bounds (linprog pkg)
        
        A = torch.hstack([b, W])
        bound = torch.hstack([bound_for_lower, bound_for_higher])
        
        A_list.append(A)
        bound_list.append(bound)

    return torch.vstack(A_list), torch.hstack(bound_list)


def optimize(c, A_ub, b_ub, A_eq, b_eq, l, u, verbose=False):

    c = c.cpu().numpy()
    A_ub, b_ub = A_ub.cpu().numpy(), b_ub.cpu().numpy()
    A_eq, b_eq = A_eq.cpu().numpy(), b_eq.cpu().numpy()

    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(l, u))

    if verbose:
        print(res)
    
    return res


# ************************** #
# *** MY FUNCTIONS (JC ) *** #
# ************************** #


# ---------------- #
# Helper functions #
# ---------------- #


def compute_errors(net, net_approx, comp_net, 
                    sample, W, b, loss_fn="cross-entropy"):
    """
    Compute the error between net and net_approx at sample x0 in 3 different ways:
    (1) Objective value at x0
    (2) Error between net and net_approx at x0
    (3) Error computed by comp_net
    """

    # Mehtod 1 (minus sign √)
    x0 = sample.flatten().cpu().numpy()
    objective_value = -objective_fn_np(x0, W, b, loss_fn=loss_fn)

    # Mehtod 2 (minus sign √)
    logits_1 = net(sample)
    logits_2 = net_approx(sample)
    real_error = -(F.softmax(logits_1, dim=1) * F.log_softmax(logits_2, dim=1)).sum().item()

    # Mehtod 3 (minus sign √)
    comp_net_with_loss_head = LossHead(comp_net, loss_fn=loss_fn)
    computed_error = comp_net_with_loss_head(sample).item()

    return objective_value, real_error, computed_error
    

def check_shapes_consistency(A, x0, cl, cu, xl, xu, verbose=True):
    """
    Checks the consistency of dimensions for the problem inputs.
    """
    m = len(cl)  # number of constraints including nonlinear
    n = len(x0)

    assert A.shape[0] + 1 == m, "Mismatch in number of constraints (A rows + 1 != m)"
    assert A.shape[1] == n, "Mismatch in number of variables (A cols != n)"
    assert len(x0) == n, "Initial guess size incorrect"
    assert len(cl) == m and len(cu) == m, "Constraint bounds size incorrect"
    assert xl.shape[0] == n and xu.shape[0] == n, "Variable bounds size incorrect"

    if verbose:
        print("✅ Shapes and sizes consistent.")


def check_bounds_and_constraints(problem, x0, xl, xu, cl, cu, 
                                 W_1=None, b_1=None, A_reduced=None, bounds=None, 
                                 p=0.7, constr_tol=1e-8, verbose=True):
    """
    Check if x is within variable bounds and satisfies constraints.
    """
    inside_bounds = np.all(x0 >= xl) and np.all(x0 <= xu)

    if isinstance(problem, NonLinearProblem):   # IPOPT version
        c_val = problem.constraints(x0)
    elif isinstance(problem, list):             # SciPy version: list of constraints in this case
        constraints_l = problem
        c_val = np.concatenate([
            np.atleast_1d(c["fun"](x0, *c.get("args", ()))) for c in constraints_l
            ])
    elif isinstance(problem, nlopt.opt):        # NLopt version
        c_js = constraints_xj_s(x0, A_reduced, bounds)
        c_0 = constraint_xi_0(x0, W_1, b_1, p)
        c_val = np.concatenate([c_js, np.array([c_0])])

    constraints_satisfied = np.all(c_val >= cl - constr_tol) and np.all(c_val <= cu) # tolerence 1e-8

    assert inside_bounds , "❌ Constraints failed at x!"
    assert constraints_satisfied , "❌ Constraints failed at x!"

    if verbose:
        print(f"\n🔍 Bounds satisfied at x:\t {inside_bounds}")
        print("✅ Bounds checks passed.")

        print(f"\n🔍 Constraints satisfied at x:\t {constraints_satisfied}")
        print("✅ Constraints check passed.")


def check_objective_gradient(problem, x0, W=None, b=None, loss_fn="cross-entropy",
                             tol=1e-4, eps=1e-6, verbose=True):
    """
    Check the gradient of the objective function via finite differences.
    """


    n = len(x0)
    if isinstance(problem, nlopt.opt):      # NLopt version
        f0 = objective_fn_np(x0, W, b, loss_fn=loss_fn)
    else:
        f0 = problem.objective(x0)

    # Finite difference gradient
    grad_fd = np.zeros(n)
    for i in range(n):
        x_eps = x0.copy()
        x_eps[i] += eps
        if isinstance(problem, nlopt.opt):  # NLopt version
            f_eps = objective_fn_np(x_eps, W, b, loss_fn=loss_fn)
        else:
            f_eps = problem.objective(x_eps)
        grad_fd[i] = (f_eps - f0) / eps

    if isinstance(problem, nlopt.opt):      # NLopt version
        grad_analytic = grad_fn_np(x0, W, b, loss_fn=loss_fn)
    else:
        grad_analytic = problem.gradient(x0)

    max_diff = np.max(np.abs(grad_analytic - grad_fd))
    assert max_diff < tol, f"❌ Gradient check failed! Max diff = {max_diff:.3e}"

    if verbose:
        print(f"\n🔍 Max difference between analytic and FD gradient: {max_diff:.3e}")
        print("✅ Gradient check passed.")


def check_constraint_jacobian(problem, x0, tol=1e-4, eps=1e-6, verbose=True):
    """
    Check the Jacobian of the constraint function via finite differences.
    """
    c0 = problem.constraints(x0)
    m = len(c0)
    n = len(x0)

    # Finite difference Jacobian
    jac_fd = np.zeros((m, n))
    for i in range(n):
        x_eps = x0.copy()
        x_eps[i] += eps
        c_eps = problem.constraints(x_eps)
        jac_fd[:, i] = (c_eps - c0) / eps

    # Analytical Jacobian from sparse representation
    jac_sparse = problem.jacobian(x0)
    jac_rows, jac_cols = problem.jacobianstructure()
    jac_dense = np.zeros((m, n))
    for val, r, c in zip(jac_sparse, jac_rows, jac_cols):
        jac_dense[r, c] = val

    # Compare
    diff = np.abs(jac_dense - jac_fd)
    max_diff = diff.max()
    assert max_diff < tol, f"❌ Jacobian check failed! Max diff = {max_diff:.3e}"

    if verbose:
        print(f"\n🔍 Max difference between analytic and FD Jacobian: {max_diff:.3e}")
        print("✅ Jacobian check passed.")


def check_predictions_consistency(x0, comp_net, verbose=True):
    """Check predictions of x0 predicted by the original and apporximated networks."""

    pred = comp_net(torch.tensor(x0).reshape((1,-1))).reshape(-1)
    half = len(pred) // 2
    class_1 = torch.argmax(pred[:half]).item()
    class_2 = torch.argmax(pred[half:]).item()
    assert class_1 == class_2 , "❌ Constraints failed at x!"

    if verbose:
        print(f"\n🔍 Predictions of x by original and approximated nets:\t {class_1, class_1}")
        print("✅ Predictions check passed.")


def check_objective_value(solution, obj_value, net, net_approx, comp_net, 
                          W, b, loss_fn="cross-entropy", verbose=True):
    """
    Checks consistency of the optimal objective value.
    """
    solution_pt = torch.from_numpy(solution).float().reshape(1, 28, 28)
    solution_pt = solution_pt.double()
    obj_value_1 = compute_errors(net, net_approx, comp_net, 
                                solution_pt, W, b, loss_fn=loss_fn)[0]
    assert np.isclose(obj_value_1, obj_value, rtol=1e-5, atol=1e-8), "❌ Objective value not consistent!"

    if verbose:
        print(f"\n🔍 Ojective values (x2):\t {obj_value_1, obj_value}")
        print("✅ Consistency check passed.")


# ------------- #
# Preliminaries #
# ------------- #


def softmax(x, axis=1):
    """Stable softmax implementation in NumPy. Uses a substracting x_max trick."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def log_softmax(x, axis=1):
    """Log-softmax for numerical stability in NumPy."""
    x_max = np.max(x, axis=axis, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return x - logsumexp


def compute_loss(logits_1, logits_2, loss_fn="cross-entropy"):
    """
    Compute cross-entropy or another custom loss between two NumPy arrays of logits.
    
    Args:
        logits_1 (np.ndarray): Shape (batch_size, num_classes)
        logits_2 (np.ndarray): Same shape
        loss_fn (str or callable): "cross-entropy" or a custom NumPy-compatible function

    Returns:
        np.ndarray: Loss values per sample (shape: (batch_size,))
    """
    if loss_fn == "cross-entropy":
        probs_1 = softmax(logits_1, axis=0)           # axis = 0 for 1D inputs
        log_probs_2 = log_softmax(logits_2, axis=0)   # axis = 0 for 1D inputs
        return -np.sum(probs_1 * log_probs_2, axis=0) # axis = 0 for 1D inputs
    else:
        return loss_fn(logits_1, logits_2)


def compute_loss_torch(logits_1, logits_2, loss_fn="cross-entropy"):
    """Torch version of the `compute_loss` function. Necessary for computing gradients."""
    if loss_fn == "cross-entropy":
        probs_1 = torch.softmax(logits_1, dim=0)
        log_probs_2 = torch.log_softmax(logits_2, dim=0)
        return -torch.sum(probs_1 * log_probs_2)
    else:
        raise NotImplementedError("Only cross-entropy is implemented in torch.")


# ------------ #
# Coefficients #
# ------------ #


def objective_coeff(comp_net, sample, mode="np"):
    """
    Get shortcut weights and biases necessary for the computation of the objective function.
    These shortcut weights are not mentioned in the paper.
    They are used to compute the outputs xi_j of the comparison network.
    """

    assert sample.shape[0] == 1 # one sample in a batch

    saturations = eval_one_sample(comp_net, sample)                     # get saturation
    target_net = squeeze_network(prune_network(comp_net, saturations))  # squezze network 

    W = target_net[-1].weight.data
    b = target_net[-1].bias.data

    if mode == "np":
        W = W.detach().cpu().numpy().astype(np.float64) # to numpy
        b = b.detach().cpu().numpy().astype(np.float64) # to numpy

    half = W.shape[0] //2       # total output size (e.g., 20)

    W_1 = W[:half, :].numpy() if mode != "np" else W[:half, :]  # first half weights for 'constraint_xi_0'
    # W_2 = W[half:, :]           # second half weights

    b_1 = b[:half].numpy() if mode != "np" else b[:half]        # first half biases for 'constraint_xi_0'
    # b_2 = b[half:]              # second half biases

    return W, b, W_1, b_1


def constraints_coeff(comp_net, sample):
    """
    Taken from Petra...

    Compute the shortcut weights and biases necessary for building the constraints.
    Shortcut weights and biases associated to saturated and unsaturated units should be W_ji and -W_ji, resp.,
    and are all associated with non-positive constraints (cf. Eq. (7)-(12)).
    NOTE: We multiply them by -1 to handle to handle non-negative constraints (required by solver).
    """
        # extract the sequential 
    #    net = next(iter(net.children())) NO NEED FOR COMPNET
    assert isinstance(comp_net, nn.Sequential)
    
    saturations = eval_one_sample(comp_net, sample)

    A_list = []
    bound_list = []

    # Compute the shortcut weights of each layer j (squeeze the subnet up to layer j) => W_ji and W_j0
    for i, saturation in enumerate(saturations):
        subnet = get_subnetwork(comp_net, i)
        if i == 0:
            target = subnet
        else:
            target = squeeze_network(prune_network(subnet, saturations[:i]))

        W = target[-1].weight.data
        b = target[-1].bias.data

        # saturated (S ~ False) and unsaturated (U ~ True) weights and biases
        W_S = W[torch.logical_not(saturation).flatten()]
        b_S = b[torch.logical_not(saturation).flatten()].reshape(-1, 1)
        W_U = W[saturation.flatten()]
        b_U = b[saturation.flatten()].reshape(-1, 1)
        
        W_minus = torch.vstack([W_U, -1*W_S]) # inverted signs for non-negative constraints
        b_minus = torch.vstack([b_U, -1*b_S]) # inverted signs for non-negative constraints
        
        A = torch.hstack([b_minus, W_minus])
        
        A_list.append(A)

    A = A.detach().cpu().numpy().astype(np.float64) # torch to numpy

    # Fix x[0] = 1
    A_reduced = A[:, 1:]    # shape (m-1, n)
    bounds = A[:, 0]        # shape (m-1,)

    return A_reduced, bounds


# --------- #
# Objective #
# --------- #


def objective_fn_np(x, W, b, loss_fn="cross-entropy"):
    """Compute non-linear objective function in numpy: Eq. (7) in short document."""

    # Perform matrix-vector multiplication (send sample to squeeze network)
    logits_all = W @ x + b
    nb_classes = len(logits_all) // 2
    logits_1, logits_2 = logits_all[:nb_classes], logits_all[nb_classes:] 

    # Compute objective as function of the inputs x_i's
    # Add "minus" sign for minimization instead of maximization
    objective = -compute_loss(logits_1, logits_2, loss_fn=loss_fn)

    return objective


def objective_fn_torch(x, W, b, loss_fn="cross-entropy"):
    """Compute non-linear objective function in numpy: Eq. (7) in short document, in PyTorch."""

    # Perform matrix-vector multiplication (send sample to squeeze network)
    logits_all = W @ x + b
    nb_classes = len(logits_all) // 2
    logits_1, logits_2 = logits_all[:nb_classes], logits_all[nb_classes:] 

    # Compute objective as function of the inputs x_i's
    # Add "minus" sign for minimization instead of maximization
    objective = -compute_loss_torch(logits_1, logits_2, loss_fn=loss_fn)

    return objective


def objective_fn_nlopt(W, b, loss_fn="cross-entropy", verbose=True):
    iteration = {"count": 0}  # mutable counter

    def nlopt_objective(x, grad):
        iteration["count"] += 1

        # Compute objective value
        val = objective_fn_np(x, W, b, loss_fn=loss_fn)

        # Compute gradient if needed
        if grad.size > 0:
            grad[:] = grad_fn_np(x, W, b, loss_fn=loss_fn)

        # Conditional printing
        if verbose:
            print(f"[NLopt Iter {iteration['count']:>4}] Objective = {val:.6f}")

        return val

    return nlopt_objective


def dummy_objective_fn_np(x, W, b, loss_fn="cross-entropy"):
    """Compute non-linear objective function in numpy: Eq. (7) in short document."""

    # Perform matrix-vector multiplication (send sample to squeeze network)
    logits_all = W @ x + b
    nb_classes = len(logits_all) // 2
    logits_1, logits_2 = logits_all[:nb_classes], logits_all[nb_classes:] 

    # Compute objective as function of the inputs x_i's
    # Add "minus" sign for minimization instead of maximization
    # objective = -compute_loss(logits_1, logits_2, loss_fn=loss_fn)
    objective = np.sum(x)

    return objective


def grad_fn_np_old(x, WW, bb, loss_fn="cross-entropy"):
    """
    Analytic computation of gradient in numpy. To be used in conjuntion with objective_fn_np.
    `loss_fn` param mandatory for consistency reasons, although not used.
    """
    x = x.reshape(-1, 1)

    half = WW.shape[0] // 2 # total output size (e.g., 20)
    W = WW[:half, :]
    W_tilde = WW[half:, :]
    b = bb[:half]
    b_tilde = bb[half:]

    xi = W @ x + b.reshape(-1, 1)           # shape: (C, 1)
    xi_tilde = W_tilde @ x + b_tilde.reshape(-1, 1)

    y = softmax(xi, axis=0)                # shape: (C, 1)
    y_tilde = softmax(xi_tilde, axis=0)

    # Grad of log(softmax): shape (C, D)
    grad_log_y_tilde = W_tilde - (y_tilde.T @ W_tilde) # MY FIX: .T added to y_tilde

    # Grad of y_k: softmax jacobian times W
    softmax_jacobian = np.diagflat(y) - y @ y.T
    grad_y_log_y_tilde = softmax_jacobian @ np.log(y_tilde + 1e-12)
    grad_part_1 = grad_y_log_y_tilde.T @ W  # shape (1, D)

    # Grad part 2
    grad_part_2 = (y.T @ grad_log_y_tilde)  # shape: (1, D)

    gradient = (grad_part_1 + grad_part_2).squeeze()

    return gradient  # shape: (D,)


def grad_fn_np(x, WW, bb, loss_fn="cross-entropy"):
    """
    My gradient, according to Jiri's Eq (22).
    Analytic computation of gradient in numpy. To be used in conjuntion with objective_fn_np.
    `loss_fn` param mandatory for consistency reasons, although not used.
    """
    x = x.reshape(-1, 1)

    half = WW.shape[0] // 2 # total output size (e.g., 20)
    W = WW[:half, :]
    W_tilde = WW[half:, :]
    b = bb[:half]
    b_tilde = bb[half:]

    xi = W @ x + b.reshape(-1, 1)
    xi_tilde = W_tilde @ x + b_tilde.reshape(-1, 1)

    y = softmax(xi, axis=0)                # shape: (C, 1)
    y_tilde = softmax(xi_tilde, axis=0)
    L = compute_loss(xi, xi_tilde, loss_fn=loss_fn)

    term_1 = W_tilde.T @ (y_tilde - y)
    term_2 = W.T @ (y * (np.log(y_tilde + 1e-12) + L))
    gradient = -(term_1 - term_2).squeeze() # minus sign for minimization instead of maximization

    return gradient  # shape: (D,)


def grad_fn_torch(x, WW, bb, loss_fn="cross-entropy"):
    """
    Analytic gradient calculation in PyTorch, converted from your numpy code.
    """
    x = x.view(-1, 1)  # ensure shape (D,1)

    half = WW.shape[0] // 2
    W = WW[:half, :]
    W_tilde = WW[half:, :]
    b = bb[:half].view(-1, 1)
    b_tilde = bb[half:].view(-1, 1)

    xi = W @ x + b       # shape: (C,1)
    xi_tilde = W_tilde @ x + b_tilde

    y = F.softmax(xi, dim=0)                # (C,1)
    y_tilde = F.softmax(xi_tilde, dim=0)

    # Compute L (loss) -- you need to implement this in torch
    L = compute_loss_torch(xi, xi_tilde, loss_fn=loss_fn)  # same as your compute_loss but in torch

    term_1 = W_tilde.T @ (y_tilde - y)
    term_2 = W.T @ (y * (torch.log(y_tilde + 1e-12) + L))

    gradient = -(term_1 - term_2).squeeze()

    return gradient



# def objective_fn_torch(x_np, W, b, loss_fn="cross-entropy"):
#     """Torch-compatible objective function for use with SciPy."""
#     # Ensure x is a differentiable tensor
#     x_torch = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)

#     # If W or b are already torch tensors, skip re-wrapping
#     if not torch.is_tensor(W):
#         W = torch.tensor(W, dtype=torch.float32)
#     if not torch.is_tensor(b):
#         b = torch.tensor(b, dtype=torch.float32)

#     logits_all = W @ x_torch + b
#     nb_classes = logits_all.shape[0] // 2
#     logits_1, logits_2 = logits_all[:nb_classes], logits_all[nb_classes:]

#     # Loss and backward
#     loss = compute_loss_torch(logits_1, logits_2, loss_fn)
#     obj = -loss
#     obj.backward()

#     return obj.item(), x_torch.grad.detach().numpy()


# def objective_fn(x, W, b, loss_fn="cross-entropy"):
#     obj, _ = objective_fn_torch(x, W, b, loss_fn)
#     return obj


# def grad_fn(x, W, b, loss_fn="cross-entropy"):
#     _, grad = objective_fn_torch(x, W, b, loss_fn)
#     return grad


# ----------- #
# Constraints #
# ----------- #


def constraint_xi_0(x, W_1, b_1, p=0.7):
    """Compute constraint xi_0 ≤ 0 Eq. (8) in short document."""
    logits = W_1 @ x + b_1
    xi_c = np.max(logits)
    xi_0 = np.sum(np.exp(logits - xi_c)) - 1/p

    return -xi_0  # flip sign for non-negative constraint


def dummy_constraint_xi_0(x, W_1, b_1, p=0.7):
    return 1.0  # Always satisfied


def constraint_xi_0_torch(x, W_1, b_1, p=0.7):
    """Compute constraint xi_0 ≤ 0 Eq. (8) in short document."""
    logits = W_1 @ x + b_1            # shape: (C,)
    xi_c = torch.max(logits)
    xi_0 = torch.sum(torch.exp(logits - xi_c)) - 1.0 / p

    return -xi_0  # flipped sign for non-negative constraint


def jac_constraint_xi_0(x, W_1, b_1, p=0.7):
    """
    Compute the Jacobian (gradient) of the constraint:
    xi_0(x) = sum_j (e^{xi_j - xi_c}) - 1/p
    Jiri's Eq. (23).

    Returns:
        grad: ndarray of shape (D,)
    """
    W = W_1

    xi = W @ x + b_1
    c = np.argmax(xi)
    xi_c = xi[c]

    exp_terms = np.exp(xi - xi_c)  # shape: (C,)
    weighted_sum = W.T @ exp_terms  # shape: (D,)
    grad = weighted_sum - W[c, :] * np.sum(exp_terms)


    # Return gradient of -xi_0 (for scipy's 'ineq' form)
    return -grad


def jac_dummy_constraint_xi_0(x, W_1, b_1, p=0.7):
    return np.zeros_like(x)  # Gradient of a constant function


def constraint_xi_0_nlopt(x, grad, W_1, b_1, p=0.7):
    """
    Wrapper for constraint_xi_0 and jac_constraint_xi_0 for NLopt.
    reverse signs for constraints of the for A @ x + b ≤ 0
    """
    if grad.size > 0:
        grad[:] = -jac_constraint_xi_0(x, W_1, b_1, p)
    return -constraint_xi_0(x, W_1, b_1, p)


def constraints_xj_s(x, A_reduced, bounds):
    """
    Generates constraints of the form xi_j ≤ 0 or xi_tilde_j ≤ 0
    Eq. (8)-(12) in short document.

    The constraints are collected into a list of inequality of the form:
    A_reduced @ x >= bounds <=> A_reduced @ x - bounds >= 0
    for use in scipy.optimize.minimize.

    Args:
        A_reduced (ndarray): Array of shape (m, n) — without the fixed x[0] term.
        bounds (ndarray): Array of shape (m,).

    Returns:
        List[dict]: List of constraint dictionaries.
    """
    return A_reduced @ x + bounds


def jac_constraints_xj_s(x, A_reduced, bounds):
    """Compute Jacobian of constraints for speedup."""
    return csr_matrix(A_reduced)


# ----------- #
# NLP Classes #
# ----------- #


class NonLinearProblemOld(object):

        def __init__(self, W, b, W_1, b_1, A, bounds, p, loss_fn="cross-entropy"):
            self.W = W
            self.b = b
            self.W_1 = W_1
            self.b_1 = b_1
            self.A = A
            self.bounds = bounds
            self.p = p
            self.n = W.shape[1]  # Number of decision variables
            self.loss_fn = loss_fn

        def objective(self, x):
            return objective_fn_np(x, self.W, self.b, loss_fn=self.loss_fn)

        def gradient(self, x):
            return grad_fn_np(x, self.W, self.b, loss_fn=self.loss_fn)

        def constraints(self, x):
            linear_part = constraints_xj_s(x, self.A, self.bounds) 
            nonlinear_part = constraint_xi_0(x, self.W_1, self.b_1, self.p)
            return np.concatenate([linear_part, [nonlinear_part]]).astype(np.float64)

        def jacobian(self, x):
            jac_xj = jac_constraints_xj_s(x, self.A, self.bounds)
            jac_x0 = jac_constraint_xi_0(x, self.W_1, self.b_1, self.p)
            full_jac = vstack([jac_xj, jac_x0])  # shape: (m+1, n)
            return full_jac.data.astype(np.float64)

        # def jacobianstructure(self):
        #     A_sparse = csr_matrix(self.A)
        #     return A_sparse.nonzero()  # tuple (row_indices, col_indices)
        
        def jacobianstructure(self):
            A_sparse = csr_matrix(self.A)
            jac_nl = jac_constraint_xi_0(np.zeros(self.n), self.W_1, self.b_1, self.p)  # dummy x
            full_jac = vstack([A_sparse, jac_nl])
            return full_jac.nonzero()


class NonLinearProblem(object):

    def __init__(self, W, b, W_1, b_1, A, bounds, p, device='cpu', loss_fn="cross-entropy"):
        # Convert to torch tensors on the specified device
        self.device = device
        self.W = torch.tensor(W, dtype=torch.float64, device=device)
        self.b = torch.tensor(b, dtype=torch.float64, device=device)
        self.W_1 = torch.tensor(W_1, dtype=torch.float64, device=device)
        self.b_1 = torch.tensor(b_1, dtype=torch.float64, device=device)
        self.W_1_np = self.W_1.cpu().numpy()
        self.b_1_np = self.b_1.cpu().numpy()
        self.A = A

        self.loss_fn = loss_fn

        self.A_sparse = csr_matrix(A)
        self.A_sparse.sort_indices()  # Ensures structure and values align
        self.jac_structure_indices = self.A_sparse.nonzero()
        self.jac_values = self.A_sparse.data.astype(np.float64)

        self.bounds = bounds
        self.p = p
        self.n = W.shape[1]

        rows_lin, cols_lin = self.jac_structure_indices
        nonlinear_row = np.full(self.n, self.A.shape[0], dtype=np.int32)
        nonlinear_col = np.arange(self.n, dtype=np.int32)
        self.row_indices = np.concatenate([rows_lin.astype(np.int32), nonlinear_row])
        self.col_indices = np.concatenate([cols_lin.astype(np.int32), nonlinear_col])


    def objective(self, x_np):
        # Convert input to torch tensor with gradient tracking
        x = torch.tensor(x_np, dtype=torch.float64, device=self.device, requires_grad=True)

        obj = objective_fn_torch(x, self.W, self.b, loss_fn=self.loss_fn)
        return obj.item()  # Return as Python float for IPOPT

    # def gradient(self, x_np):
    #     x = torch.tensor(x_np, dtype=torch.float64, device=self.device, requires_grad=True)

    #     obj = objective_fn_torch(x, self.W, self.b, loss_fn=self.loss_fn)
    #     grad, = torch.autograd.grad(obj, x, create_graph=False)
    #     return grad.detach().cpu().numpy()
    
    def gradient(self, x_np):
        x = torch.tensor(x_np, dtype=torch.float64, device=self.device)
        grad = grad_fn_torch(x, self.W, self.b, loss_fn=self.loss_fn)
        return grad.detach().cpu().numpy()

    # def hessian(self, x_np):
    #     # Optional: add if IPOPT or your solver supports Hessian evaluation
    #     x = torch.tensor(x_np, dtype=torch.float64, device=self.device, requires_grad=True)
    #     # Use torch.autograd.functional.hessian (PyTorch 1.5+)
    #     hess = torch.autograd.functional.hessian(
    #         lambda x_: objective_fn_torch(x_, self.W, self.b, loss_fn=self.loss_fn),
    #         x
    #     )
    #     return hess.detach().cpu().numpy()

    def constraints(self, x):
        # Keep your numpy constraints as before, since constraints functions appear numpy-based
        # start = time.time()
        linear_part = constraints_xj_s(x, self.A, self.bounds)
        nonlinear_part = constraint_xi_0(x, self.W_1_np, self.b_1_np, self.p)
        return np.concatenate([linear_part, [nonlinear_part]]).astype(np.float64)

    def jacobian(self, x):
        # start = time.time()
        jac_x0 = jac_constraint_xi_0(x, self.W_1_np, self.b_1_np, self.p).astype(np.float64)
        return np.concatenate([self.jac_values, jac_x0])

    
    def jacobianstructure(self):
        # start = time.time()
        # Nonlinear constraint: dense row at the bottom (last row)
        return self.row_indices, self.col_indices



# # --------------------- #
# # Optimization function #
# # --------------------- #


# # *** Scipy *** #

# def compute_error_scipy(net, net_approx, comp_net, sample, output_dir, p=0.7,
#                         nb_constraints="all", start=0, end=1, loss_fn="cross-entropy",
#                         device="cpu", verbose=False):
                
#         net = copy.deepcopy(net)                # deep copy for safety reasons
#         net_approx = copy.deepcopy(net_approx)  # deep copy for safety reasons
#         comp_net = copy.deepcopy(comp_net)      # deep copy for safety reasons

#         sample = sample.to(device).double()

#         # Start timer
#         start_time = time.time()
        
#         # Non-Linear Problem (NLP)
#         # Coefficients
#         W, b, W_1, b_1 = objective_coeff(comp_net, sample, mode="np")
#         A_reduced, bounds = constraints_coeff(comp_net, sample)
#         if nb_constraints != "all":
#             A_reduced = A_reduced[:nb_constraints]
#             bounds = bounds[:nb_constraints]
        
#         m = A_reduced.shape[0] + 1
#         n = A_reduced.shape[1]

#         # Bounds
#         # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
#         xl = np.ones(n, dtype=np.float64)*(-0.5)
#         xu = np.ones(n, dtype=np.float64)*2.9
#         # Constraints' bounds: [0, ∞)
#         cl = np.concatenate([np.zeros(m - 1), [0.0]])
#         cu = np.concatenate([np.full(m - 1, np.inf), [np.inf]])

#         # Constraint in dict form for minimize()
#         constraints = [
#             {
#             'type': 'ineq', 
#             'fun': constraint_xi_0,
#             'jac' : jac_constraint_xi_0, # provide analytic Jacobian
#             'args': (W_1, b_1, p)
#             },
#             {
#             'type': 'ineq', 
#             'fun': constraints_xj_s, 
#             'jac': jac_constraints_xj_s, # provide analytic Jacobian
#             'args': (A_reduced, bounds)
#             }
#                     ]

#         # Bounds
#         # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
#         input_bounds = Bounds([-0.5]*W.shape[1], [2.9]*W.shape[1])

#         # Initial guess: sample itself
#         x0 = sample.flatten().cpu().numpy()

#         # Run minimization
#         method = 'trust-constr' # 'trust-constr' (better but slower), 'SLSQP'

#         options = {
#             'maxiter': 10000,
#             'disp': True,
#             'sparse_jacobian' : True, # improves a lot!
#             'xtol' : 1e-6,
#             'gtol': 1e-6,           # Gradient norm tolerance
#         }

#         iteration = [0]

#         def callback_fn(xk, state=None):
#             iteration[0] += 1
#             if verbose:
#                 print(".", end="", flush=True)

#         res = minimize(
#                     objective_fn_np, x0, args=(W, b, loss_fn), # objective
#                     jac=grad_fn_np,             # gradients: (better without??? not clear...)
#                     bounds=input_bounds,        # bounds 
#                     constraints=constraints,    # constraints
#                     method=method,
#                     # hess=lambda x, *args: np.zeros((len(x), len(x))),
#                     options=options,
#                     callback=callback_fn
#                     )
        
#         # Compute sample error (3 methods)
#         objective_value, real_error, computed_error = compute_errors(net, net_approx, comp_net, sample, 
#                                                                      W, b, loss_fn=loss_fn)

#         # results: error_1, error_2, error_3 (should coincide) and polytope error
#         with open(f"{output_dir}/results_{start}_{end}_nlp.csv", "a") as f:
#             f.write(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-res.fun}\n")

#         if verbose:
            
#             print("\n-----------------------\n")

#             check_shapes_consistency(A_reduced, x0, cl, cu, xl, xu, verbose)
#             check_bounds_and_constraints(constraints, x0, xl, xu, cl, cu, constr_tol=1e-6, verbose=verbose)

#             print("\n-----------------------")


#             print("\nErrors at x0 (1,2,3) and maximal error (4)")
#             print(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-res.fun}")

#             print("\n✅ Optimal solution:", res.x.shape)
#             print("Objective value:", res.fun)
#             check_bounds_and_constraints(constraints, res.x, xl, xu, cl, cu, constr_tol=1e-6, verbose=verbose)

#             check_objective_value(res.x, -res.fun, 
#                                 net, net_approx, comp_net, 
#                                 W, b, loss_fn=loss_fn, verbose=verbose)
            
#             check_predictions_consistency(x0, comp_net)
#             check_predictions_consistency(res.x, comp_net)

#             end_time = time.time()
#             elapsed_time = end_time - start_time    
#             print(f"\nOptimization time: {elapsed_time:.4f} seconds")

#             print("\n-----------------------\n")
            
#             # check gradient
#             grad_err = check_grad(objective_fn_np, grad_fn_np, x0, W, b, loss_fn)
#             print("🔍 Gradient error:", grad_err)

#             # check jacobians
#             def wrapper_1(x):
#                 return constraint_xi_0(x, W_1, b_1, p)
#             def wrapper_2(x):
#                 return constraints_xj_s(x, A_reduced, bounds)
            
#             J_numeric_1 = approx_derivative(wrapper_1, x0)
#             J_analytic_1 = jac_constraint_xi_0(x0, W_1, b_1, p)
#             print("🔍 Jacobian #1 error:", np.max(np.abs(J_numeric_1 - J_analytic_1)))

#             J_numeric_2 = approx_derivative(wrapper_2, x0)
#             J_analytic_2 = jac_constraints_xj_s(x0, A_reduced, bounds)
#             print("🔍 Jacobian #2 error:", np.max(np.abs(J_numeric_2 - J_analytic_2)))

#         print("\n-----------------------\n")


# # *** IPOPT *** #


# def compute_error_ipopt(net, net_approx, comp_net, sample, output_dir, p=0.7,
#                         nb_constraints="all", start=0, end=1, loss_fn="cross-entropy", 
#                         device="cpu", tol=1e-6, verbose=False):

#     net = copy.deepcopy(net)                # deep copy for safety reasons
#     net_approx = copy.deepcopy(net_approx)  # deep copy for safety reasons
#     comp_net = copy.deepcopy(comp_net)      # deep copy for safety reasons

#     sample = sample.to(device).double()

#     # Start timer
#     start_time = time.time()
    
#     # # Non-Linear Problem (NLP)
#     # Ccoefficients
#     W, b, W_1, b_1 = objective_coeff(comp_net, sample, mode="np")
#     A_reduced, bounds = constraints_coeff(comp_net, sample)
#     if nb_constraints != "all":
#         A_reduced = A_reduced[:nb_constraints]
#         bounds = bounds[:nb_constraints]

#     # Initial guess: sample itself
#     x0 = sample.flatten().cpu().numpy()

#     m = A_reduced.shape[0] + 1
#     n = A_reduced.shape[1]

#     # Bounds
#     # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
#     xl = np.ones(n, dtype=np.float64)*(-0.5)
#     xu = np.ones(n, dtype=np.float64)*2.9

#     # Constraints' bounds: [0, ∞)
#     cl = np.concatenate([np.zeros(m - 1), [0.0]])
#     cu = np.concatenate([np.full(m - 1, np.inf), [np.inf]])
    
#     problem_obj = NonLinearProblem(W, b, W_1, b_1, A_reduced, bounds, p=p)

#     nlp = cyipopt.Problem(
#             n=n,    # nb of variables
#             m=m,    # nb of constraints
#             lb=xl,  # lower bounds
#             ub=xu,  # upper bounds
#             cl=cl,  # constraints lower bounds
#             cu=cu,  # constraints upper bounds
#             problem_obj=problem_obj
#         )

#     print_level = 5 if verbose==True else 1
#     nlp.add_option("print_level", print_level)
#     nlp.add_option("tol", tol)
#     nlp.add_option("hessian_approximation", "limited-memory")
#     constr_tol = tol
#     nlp.add_option("constr_viol_tol", constr_tol)

#     # Run minimization
#     solution, info = nlp.solve(x0) # solve problem

#     # Compute sample error (3 methods)
#     objective_value, real_error, computed_error = compute_errors(net, net_approx, comp_net, sample, 
#                                                                  W, b, loss_fn=loss_fn)

#     # results: error_1, error_2, error_3 (should coincide) and polytope error
#     with open(f"{output_dir}/results_{start}_{end}_nlp.csv", "a") as f:
#         f.write(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-info["obj_val"]}\n")        

#     if verbose:

#         print("\n-----------------------\n")

#         check_shapes_consistency(A_reduced, x0, cl, cu, xl, xu, verbose)
#         check_bounds_and_constraints(problem_obj, x0, xl, xu, cl, cu, constr_tol=constr_tol, verbose=verbose)
#         check_objective_gradient(problem_obj, x0, loss_fn=loss_fn, verbose=verbose)
#         check_constraint_jacobian(problem_obj, x0, verbose)
#         check_predictions_consistency(x0, comp_net)

#         print("\n-----------------------")

#         print("\nErrors at x0 (1,2,3) and maximal error (4)")
#         print(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-info["obj_val"]}")
#         print("\n✅ Optimal solution:", solution.shape)
#         print("Objective value:", info["obj_val"])
#         check_bounds_and_constraints(problem_obj, solution, xl, xu, cl, cu, constr_tol=constr_tol, verbose=verbose)
#         check_objective_value(solution, -info["obj_val"], 
#                             net, net_approx, comp_net, 
#                             W, b, loss_fn=loss_fn, verbose=verbose)
#         check_predictions_consistency(solution, comp_net)

#         print("\n-----------------------\n")


# # *** NLopt *** #


# def compute_error_nlopt(net, net_approx, comp_net, sample, output_dir, p=0.7,
#                     nb_constraints="all", start=0, end=1, loss_fn="cross-entropy", 
#                     device="cpu", nb_iter=15000, tol=1e-6, verbose=False):
            
#     net = copy.deepcopy(net)                # deep copy for safety reasons
#     net_approx = copy.deepcopy(net_approx)  # deep copy for safety reasons
#     comp_net = copy.deepcopy(comp_net)      # deep copy for safety reasons

#     sample = sample.to(device).double()

#     # Start timer
#     start_time = time.time()
    
#     # Non-Linear Problem (NLP)
#     # Coefficients
#     W, b, W_1, b_1 = objective_coeff(comp_net, sample, mode="np")
#     A_reduced, bounds = constraints_coeff(comp_net, sample)
#     if nb_constraints != "all":
#         A_reduced = A_reduced[:nb_constraints]
#         bounds = bounds[:nb_constraints]
    
#     m = A_reduced.shape[0] + 1
#     n = A_reduced.shape[1]
    
#     # Bounds
#     # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
#     xl = np.ones(n, dtype=np.float64)*(-0.5)
#     xu = np.ones(n, dtype=np.float64)*2.9
#     # Constraints' bounds: [0, ∞)
#     cl = np.concatenate([np.zeros(m - 1), [0.0]])
#     cu = np.concatenate([np.full(m - 1, np.inf), [np.inf]])

#     # Problem
#     opt = nlopt.opt(nlopt.AUGLAG, n)
#     opt.set_maxeval(nb_iter)
#     opt.set_ftol_rel(tol)
#     opt.set_xtol_rel(tol)

#     local_opt = nlopt.opt(nlopt.LD_MMA, n)
#     local_opt.set_maxeval(nb_iter)  # Try small inner loop budget
#     local_opt.set_ftol_rel(tol)
#     local_opt.set_xtol_rel(tol)
#     opt.set_local_optimizer(local_opt)

#     # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
#     opt.set_lower_bounds([-0.5]*n)
#     opt.set_upper_bounds([2.9]*n)

#     obj_fn = objective_fn_nlopt(W, b, loss_fn=loss_fn, verbose=verbose)
#     opt.set_min_objective(obj_fn)   # minimize!

#     # Constraints (linear and non-linear)
#     # A_minus x + b_minus ≤ 0       <=>   
#     # -A_reduced x - bounds ≤ 0     <=>
#     # -(-W) x - (-b) ≤ 0            <=>
#     # W x + b ≤ 0                   (eq. (7)-(12) √)
#     A_minus = -A_reduced
#     b_minus = -bounds

#     def linear_constraints_vectorized(result, x, grad):
#         if grad.size > 0:
#             grad[:] = A_minus
#         result[:] = A_minus @ x + b_minus
#         return None # nlopt requirement

#     opt.add_inequality_mconstraint(linear_constraints_vectorized, [tol] * A_minus.shape[0])
#     opt.add_inequality_constraint(lambda x, grad: constraint_xi_0_nlopt(x, grad, W_1, b_1, p=p), tol)

#     # Initial guess
#     x0 = sample.flatten().cpu().numpy()
#     x_opt = opt.optimize(x0)
#     dummy_grad = np.zeros_like(x_opt)
#     obj_val = objective_fn_nlopt(W, b, loss_fn=loss_fn, verbose=verbose)(x_opt, dummy_grad)

#     # Compute sample error (3 methods)
#     objective_value, real_error, computed_error = compute_errors(net, net_approx, comp_net, sample, 
#                                                                     W, b, loss_fn=loss_fn)

#     # results: error_1, error_2, error_3 (should coincide) and polytope error
#     with open(f"{output_dir}/results_{start}_{end}_nlp.csv", "a") as f:
#         f.write(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-obj_val}\n")        

#     if verbose:

#         print("\n-----------------------\n")

#         check_shapes_consistency(A_reduced, x0, cl, cu, xl, xu, verbose)
#         check_bounds_and_constraints(opt, x0, xl, xu, cl, cu, 
#                                         W_1=W_1, b_1=b_1, A_reduced=A_reduced, bounds=bounds, 
#                                         p=p, constr_tol=tol, verbose=verbose)
#         check_objective_gradient(opt, x0, W=W, b=b, verbose=verbose)
#         # check_constraint_jacobian(problem_obj, x0, verbose)
#         check_predictions_consistency(x0, comp_net)

#         print("\n-----------------------")

#         print("\nErrors at x0 (1,2,3) and maximal error (4)")
#         print(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-obj_val}")
#         print("\n✅ Optimal solution:", x_opt.shape)
#         print("Objective value:", obj_val)
#         check_bounds_and_constraints(opt, x_opt, xl, xu, cl, cu, 
#                                         W_1=W_1, b_1=b_1, A_reduced=A_reduced, bounds=bounds, 
#                                         p=p, constr_tol=tol, verbose=verbose)
#         check_objective_value(x_opt, -obj_val, 
#                             net, net_approx, comp_net, 
#                             W, b, loss_fn=loss_fn, verbose=verbose)
#         check_predictions_consistency(x_opt, comp_net)

#         print("\n-----------------------\n")


# # Uncomment this for testing

# if __name__ == "__main__":

#     # Parameters
#     DEVICE = "cpu"
#     print(f"Using device: {DEVICE}\n")

#     model_name = "mnist_smalldensenet_1024_50.pt"
#     bits = 16

#     start, end = 42, 44
#     output_dir = "."
        
#     # Dataset
#     test_dataset = create_dataset(mode="experiment")
#     subset_dataset = Subset(test_dataset, list(range(start, end)))

#     # Network
#     NETWORK = os.path.join("checkpoints", model_name)
#     MODEL = SmallDenseNet 
#     # LAYERS = 4
#     # INPUT_SIZE = (1, 28, 28) 
#     # N = 1 * 28 * 28 

#     net = load_network(MODEL, NETWORK, device=DEVICE)
#     net_approx = load_network(MODEL, NETWORK, device=DEVICE)
#     comp_net = create_comparing_network(net, net_approx, bits=bits, skip_magic=True)

#     # Compute errors
#     for sample, _ in tqdm(subset_dataset, desc="Processing"):

#         pass

#     # compute_error_nlopt(net, net_approx, comp_net, sample, output_dir, p=0.7,
#     #                         nb_constraints="all", start=0, end=1, loss_fn="cross-entropy", 
#     #                         device="cpu", nb_iter=15000, tol=1e-6, verbose=True)