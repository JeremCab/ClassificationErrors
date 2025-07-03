import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import Bounds, linprog, minimize, check_grad

from preprocessing import eval_one_sample, squeeze_network, prune_network, get_subnetwork, truncate_after_last_relu

TOL = 1e-8   # almost 0: to check the bounds
TOL2 = 1e-9  # almost 0: to check the bounds

def create_c(compnet, inputs):
    assert inputs.shape[0] == 1 # one sample in a batch

    # wide_inputs = torch.hstack([inputs, inputs]) TODO: delete this line

    # reduce and squeeze compnet 
    saturations = eval_one_sample(compnet, inputs)
    target_net = squeeze_network(prune_network(compnet, saturations))

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
            target = squeeze_network(prune_network(subnet, saturations[:i])) # XXX prunning can be done once for all before... and pass the pruned networks here

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


def objective_coeff(compnet, sample):
    """
    Get shortcut weights and biases necessary for the computation of the objective function.
    These shortcut weights are not mentioned in the paper.
    They are used to compute tue outputs xi_j of the comparison network.
    Note that W and b (to be used in the objective function) are returned in torch
    while W_1 and b_1 (to be used in the constraints) are returned in numpy.
    """

    assert sample.shape[0] == 1 # one sample in a batch

    saturations = eval_one_sample(compnet, sample)                     # get saturation
    target_net = squeeze_network(prune_network(compnet, saturations))  # squezze network 

    W = target_net[-1].weight.data
    b = target_net[-1].bias.data

    # W = W.detach().cpu().numpy().astype(np.float64) # to numpy
    # b = b.detach().cpu().numpy().astype(np.float64) # to numpy

    half = W.shape[0] //2       # total output size (e.g., 20)

    W_1 = W[:half, :].numpy()   # first half weights / to numpy for 'minimize'
    # W_2 = W[half:, :]         # second half weights

    b_1 = b[:half].numpy()      # first half biases / to numpy for 'minimize'
    # b_2 = b[half:]            # second half biases

    return W, b, W_1, b_1


def constraints_coeff(compnet, sample):
    """
    Taken from Petra...

    Compute the shortcut weights and biases necessary for building the contraints.
    Shortcut weights and biases associated to saturated and unsaturation units should be W_ji and -W_ji, resp.,
    and are all associated twith non-positive contraints (cf. Eq. (7)-(12)).
    We multiply them by -1 to handle non-positive contraints (required by solver).
    """
        # extract the sequential 
    #    net = next(iter(net.children())) NO NEED FOR COMPNET
    assert isinstance(compnet, nn.Sequential)
    
    saturations = eval_one_sample(compnet, sample)

    A_list = []
    bound_list = []

    # Compute the shortcut weights of each layer j (squeeze the subnet up to layer j) => W_ji and W_j0
    for i, saturation in enumerate(saturations):
        subnet = get_subnetwork(compnet, i)
        if i == 0:
            target = subnet
        else:
            target = squeeze_network(prune_network(subnet, saturations[:i])) # XXX prunning can be done once for all before... and pass the pruned networks here

        W = target[-1].weight.data
        b = target[-1].bias.data

        # unsaturated (U) and saturated (S) weights and biases
        W_U = W[torch.logical_not(saturation).flatten()]
        b_U = b[torch.logical_not(saturation).flatten()].reshape(-1, 1)
        W_S = W[saturation.flatten()]
        b_S = b[saturation.flatten()].reshape(-1, 1)

        bound_for_lower = torch.full((W_U.shape[0],), -TOL, dtype=torch.float64)
        bound_for_higher = torch.full((W_S.shape[0],), -TOL, dtype=torch.float64)
        
        W = torch.vstack([W_U, -1*W_S])
        b = torch.vstack([b_U, -1*b_S])
        
        A = torch.hstack([b, W])
        bound = torch.hstack([bound_for_lower, bound_for_higher])
        
        A_list.append(A)
        bound_list.append(bound)

    A, bounds = torch.vstack(A_list), torch.hstack(bound_list)
    A = A.detach().cpu().numpy().astype(np.float64)            # torch to numpy
    bounds = bounds.detach().cpu().numpy().astype(np.float64)  # torch to numpy

    # Fix x[0] = 1
    A_reduced = A[:, 1:]  # Shape: (6000, 784)
    adjusted_bounds = bounds - A[:, 0]  # Shape: (6000,)

    return A_reduced, adjusted_bounds


# --------- #
# Objective #
# --------- #


# def objective_fn(x, W, b, loss_fn="cross-entropy"):
#     """Compute non-linear objective function: Eq. (7) in short document."""

#     # Perform matrix-vector multiplication (send sample to squeeze network)
#     logits_all = W @ x + b
#     nb_classes = len(logits_all) // 2
#     logits_1, logits_2 = logits_all[:nb_classes], logits_all[nb_classes:] 

#     # Compute objective as function of the inputs x_i's
#     # Add "minus" sign for minimization instead of maximization
#     objective = -compute_loss(logits_1, logits_2, loss_fn=loss_fn)

#     return objective


def objective_fn_torch(x_np, W, b, loss_fn="cross-entropy"):
    """Torch-compatible objective function for use with SciPy."""
    # Ensure x is a differentiable tensor
    x_torch = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)

    # If W or b are already torch tensors, skip re-wrapping
    if not torch.is_tensor(W):
        W = torch.tensor(W, dtype=torch.float32)
    if not torch.is_tensor(b):
        b = torch.tensor(b, dtype=torch.float32)

    # Forward pass
    logits_all = W @ x_torch + b
    nb_classes = logits_all.shape[0] // 2
    logits_1, logits_2 = logits_all[:nb_classes], logits_all[nb_classes:]

    # Loss and backward
    loss = compute_loss_torch(logits_1, logits_2, loss_fn)
    obj = -loss
    obj.backward()

    return obj.item(), x_torch.grad.detach().numpy()


def objective_fn(x, W, b, loss_fn="cross-entropy"):
    obj, _ = objective_fn_torch(x, W, b, loss_fn)
    return obj


def grad_fn(x, W, b, loss_fn="cross-entropy"):
    _, grad = objective_fn_torch(x, W, b, loss_fn)
    return grad



# ----------- #
# Constraints #
# ----------- #


def constraint_xi_0(x, W_1, b_1, p=0.85):
    """Compute costraint xi_0 ≤ 0 Eq. (8) in short document."""

    logits = W_1 @ x + b_1
    xi_c = np.max(logits)
    xi_0 = np.sum(np.exp(logits - xi_c)) - 1/p

    return -xi_0  # flip sign for non-positive constraint


def constraints_xj_s(x, A_reduced, bounds):
    """
    Generates constraints of the form xi_j ≤ 0 or xi_tilde_j ≤ 0
    Eq. (8)-(12) in short document.

    The contraints are collected into a list of inequality of the form:
    A_reduced @ x >= bounds <=> A_reduced @ x - bounds >= 0
    for use in scipy.optimize.minimize.

    Args:
        A_reduced (ndarray): Array of shape (m, n) — without the fixed x[0] term.
        bounds (ndarray): Array of shape (m,).

    Returns:
        List[dict]: List of constraint dictionaries.
    """
    return A_reduced @ x - bounds




if __name__ == "__main__":


    # XXX XXX XXX Example
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocessing import LossHead, create_comparing_network, eval_one_sample
    from utils.network import load_network, SmallConvNet, SmallDenseNet
    from utils.dataset import create_dataset
    from torch.utils.data import Subset
    import matplotlib.pyplot as plt

    MODE = "debug"
    DEVICE = "cpu"
    model_name = "mnist_smalldensenet_1024_2.pt"
    bits = 16

    NETWORK = os.path.join("checkpoints", model_name)
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    # 1. Get sample (just for testing) # XXX
    test_dataset = create_dataset(mode="experiment")
    subset_dataset = Subset(test_dataset, list(range(13, 14))) # 1 arbitraty sample

    for sample, _ in subset_dataset:
        sample = sample.to(DEVICE).double()
        break


    # 2. Compute comparing network
    net = load_network(MODEL, NETWORK, device=DEVICE)
    net_approx = load_network(MODEL, NETWORK, device=DEVICE)
    # NOTE: net_approx is modified here (not obvious, discovered after investgation)!
    # XXX was skip_magic=False before
    compnet = create_comparing_network(net, net_approx, bits=bits, skip_magic=True)
    
    # 3. Solving the NLP
    # (i) Compute coefficients of the NLP
    p = 0.75
    W, b, W_1, b_1 = objective_coeff(compnet, sample)
    A_reduced, bounds = constraints_coeff(compnet, sample)

    # (ii) Wrap the constraint in dict form for minimize()
    constraints = [
        {'type': 'ineq', 'fun': constraint_xi_0, 'args': (W_1, b_1, p)},
        {'type': 'ineq', 'fun': constraints_xj_s, 'args': (A_reduced, bounds)},
                   ]

    input_bounds = Bounds([0]*W.shape[1], [1]*W.shape[1])  # inputs satisfy 0 ≤ x[i] ≤ 1

    # (iii) Initial guess (same dimension as input x)
    x0 = np.ones(W_1.shape[1]).astype(np.float64)
    
    # (iv) Run the minimization
    
    
    # # XXX to be moved...

    method = 'trust-constr'  # Solver: 'trust-constr' (better but slower) or 'SLSQP'

    options = {
        'maxiter': 10,
        'disp': True
    }

    if method == 'trust-constr':
        options['xtol'] = 1e-5
    elif method == 'SLSQP':
        options['ftol'] = 1e-5

    iteration = [0]

    def callback_fn(xk, state=None):
        iteration[0] += 1
        print(".", end="", flush=True)

    # check gradients
    if MODE == "debug":
        err = check_grad(objective_fn, grad_fn, x0, W, b, "cross-entropy")
        print("Gradient check error:", err)

    res = minimize(
                objective_fn, x0, args=(W, b, "cross-entropy"), # objective
                jac=grad_fn,                # gradients: doesn't seem to accelerate...
                bounds=input_bounds,        # bounds 
                constraints=constraints,    # contraints
                method=method,
                options=options, 
                callback=callback_fn
                )

    print("res:", res)
    print("Objective value:", res.fun)
    print("Xi_0 constraint value (should be >= 0):", constraint_xi_0(res.x, W_1, b_1, p))
    print("Xi_j constraints values (should be >= 0):", constraints_xj_s(res.x, A_reduced, bounds))



    # 4. Sanity checks
    # NOTE: The Numpy conversion and numerical stability computation trick introduces tiny errors...
    if MODE == "debug":

        print("\nSanity check...")
        
        # Error: mehtod 1 (minus sign √)
        x = sample.flatten().detach().cpu().numpy().astype(np.float64) # XXX Just for testing!
        objective_value = -objective_fn(x, W, b, loss_fn="cross-entropy") # (minus) objective value at the sample <=> error
        print("Objective value:", objective_value)
        
        # Error: mehtod 2 (minus sign √)
        logits_1 = net(sample)
        logits_2 = net_approx(sample)
        real_error = -(F.softmax(logits_1, dim=1) * F.log_softmax(logits_2, dim=1)).sum().item()

        # Error: mehtod 3 (minus sign √)
        compnet_with_loss_head = LossHead(compnet, loss_fn="cross-entropy")
        computed_error = compnet_with_loss_head(sample).item()

        # Checks
        print("Errors:", real_error, computed_error, objective_value)
        assert abs(real_error - computed_error) < TOL
        assert abs(computed_error - objective_value) < TOL
        assert abs(objective_value - real_error) < TOL
    
