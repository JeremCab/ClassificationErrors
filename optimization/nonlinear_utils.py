import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog, minimize

from preprocessing import eval_one_sample, squeeze_network, prune_network, get_subnetwork

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


def objective_coeff(compnet, sample):
    """Get shortcut weights are biases necessary for the computation of the objective function."""

    assert sample.shape[0] == 1 # one sample in a batch

    saturations = eval_one_sample(compnet, sample)                     # get saturation
    target_net = squeeze_network(prune_network(compnet, saturations))  # squezze network 

    W = target_net[-1].weight.data
    b = target_net[-1].bias.data

    W = W.detach().cpu().numpy() # to numpy
    b = b.detach().cpu().numpy() # to numpy

    half = W.shape[0] //2             # total output size (e.g., 20)

    W_1 = W[:half, :]                 # first half weights
    W_2 = W[half:, :]                 # second half weights

    b_1 = b[:half]                   # first half biases
    b_2 = b[half:]                   # second half biases

    return W, b, W_1, b_1, W_2, b_2


def objective_fn(x, W, b, loss_fn="cross-entropy"):
    """Compute non-linear objective function: Eq. (7) in short document."""

    # Perform matrix-vector multiplication (send sample to squeeze network)
    logits_all = W @ x + b
    nb_classes = len(logits_all) // 2
    logits_1, logits_2 = logits_all[:nb_classes], logits_all[nb_classes:] 

    # Compute objective as function of the inputs x_i's
    objective = compute_loss(logits_1, logits_2, loss_fn=loss_fn)

    return objective


def constraint_xi_0(x, W_1, b_1, p=0.85):
    """Compute costraint xi_0 ≤ 0 Eq. (8) in short document."""

    logits = W_1 @ x + b_1
    xi_c = np.max(logits)
    xi_0 = np.sum(np.exp(logits - xi_c)) - 1/p

    return -xi_0  # flip sign for negative constraint



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

    # 2. Compute comparing etwork
    net = load_network(MODEL, NETWORK, device=DEVICE)
    net_approx = load_network(MODEL, NETWORK, device=DEVICE)

    # NOTE: net_approx is modified here (not obvious, discovered after investgation)!
    compnet = create_comparing_network(net, net_approx, bits=bits, skip_magic=True) # XXX was skip_magic=False before
    compnet_with_loss_head = LossHead(compnet, loss_fn="cross-entropy")

    # 3. Define non-linear problem
    x = sample.flatten().detach().cpu().numpy() # XXX Just for testing! Should be removed

    # 3. Coefficients
    W, b, W_1, b_1, W_2, b_2 = objective_coeff(compnet, sample)
    p = 0.85

    # 4. Solving the NLP
    # Initial guess (same dimension as input x)
    x0 = np.zeros(W_1.shape[1])

    # Wrap the constraint in dict form for minimize()
    constraints = [{'type': 'ineq', 'fun': constraint_xi_0, 'args': (W_1, b_1, p)}]

    # Run the minimization
    res = minimize(objective_fn, x0, args=(W, b, "cross-entropy"), constraints=constraints, method='SLSQP')

    print("Objective value:", res.fun)
    print("Constraint value (should be >= 0):", constraint_xi_0(res.x, W_1, b_1, p))

    # Sanity cheks
    # NOTE: The Numpy conversion and associated numerical stability computation trick 
    # introduces tiny errors...
    if MODE == "debug":
        print("\nSanity check...")
        objective_value = objective_fn(x, W, b, loss_fn="cross-entropy")
        print("Objective value:", objective_value)
        logits_1 = net(sample)
        logits_2 = net_approx(sample)
        logits_3 = compnet(sample)

        real_error = -(F.softmax(logits_1, dim=1) * F.log_softmax(logits_2, dim=1)).sum().item()
        computed_error = compnet_with_loss_head(sample).item()
        # print("Errors:", real_error, computed_error, objective_value)
        assert abs(real_error - computed_error) < TOL
        assert abs(computed_error - objective_value) < TOL
        assert abs(objective_value - real_error) < TOL
    


    

    

    # XXX XXX XXX


    # def objective(x):
    #     return np.sin(x[0]) + x[1]**2   # non-convex objective
    
    def objective(x):
        return np.sin(x[0]) + x[1]**2   # non-convex objective

    def constraint1(x):
        return x[0] + x[1] - 1          # convex constraint

    x0 = [0.5, 0.5]
    constraints = [{'type': 'ineq', 'fun': constraint1}]

    res = minimize(objective, x0, constraints=constraints, method='SLSQP')  # or 'trust-constr'
    print(res.x, res.fun)
