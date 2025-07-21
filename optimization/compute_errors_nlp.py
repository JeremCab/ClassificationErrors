import sys
import os

from tqdm import tqdm
import argparse
import yaml

import copy

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

# Don't need this line since "export PYTHONPATH=$(pwd)" in train_network.sh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from propagate_intervals.train import parse_config
from preprocessing import LossHead, create_comparing_network, eval_one_sample
from utils.network import load_network, SmallConvNet, SmallDenseNet
from utils.dataset import create_dataset
from linear_utils import create_c, create_upper_bounds, optimize
from nonlinear_utils import *
from linear_utils import TOL, TOL2


def check_upper_bounds(A, b, input_1, input_2, verbose=False):
    """
    This function checks the inequalities contraints A @ x <= ε of the linear program (LP)
    for both the original input sample (`input_1`) and the solution from the LP (`input_2`).

    Args:
        A (torch.Tensor): The constraint matrix of shape (m, n).
        b (torch.Tensor): The constraint vector of shape (m,).
        input_1 (torch.Tensor): The original input sample.
        input_2 (torch.Tensor or np.ndarray): The flattened solution of the LP.
        verbose (bool): If True, prints diagnostic information when constraints are violated.

    Raises:
        AssertionError: If any of the inequality constraints are violated for input_1 or input_2.
    """

    A = A.cpu()

    input_1 = input_1.cpu() # put sample (input_1) back to cpu
    # Note that the sol of the LP (input_2) is already on cpu.
    
    input_1 = torch.hstack([torch.tensor(1), input_1.reshape(-1)])   # add 1 at the beginning for bias
    input_2 = torch.hstack([torch.tensor(1), torch.tensor(input_2)]) # add 1 at the beginning for bias

    result_1 = A @ input_1
    assert torch.all(result_1 <= TOL + TOL2)
    
    result_2 = A @ input_2 
    assert torch.all(result_2 <= TOL + TOL2)
    
    wrong_indexes = torch.logical_not(result_2 <= TOL + TOL2)      # mask of the violated constraints
    
    if verbose:
        print("********************")
        print(A.shape)
        print(input_1.shape)
        print(input_2.shape)
        print("Check upper bounds 1: ", torch.all(result_1 <= TOL + TOL2).item())
        print("Check upper bounds 2: ", torch.all(result_2 <= TOL + TOL2 ).item())
        print("Nb of wrong indices:", wrong_indexes.sum().item())
        print("Wrong indices:", result_2[wrong_indexes])
        print("********************")


def check_saturations(net, input_1, input_2, verbose=False):
    """
    Verifies that the network produces the same activation (saturation) pattern
    for both the original data sample (input_1) and the LP solution (input_2).

    Args:
        net (torch.nn.Module): The trained neural network.
        input_1 (torch.Tensor): Original data sample.
        input_2 (array-like): LP solution (flattened), to be reshaped into data format.
        verbose (bool): If True, prints whether the saturation pattern matches.

    Raises:
        AssertionError: If the saturation patterns do not match.
    """

    device = next(net.parameters()).device

    # Rehape sol of the LP to sample size
    input_2 = torch.tensor(input_2).reshape(1, 1, 28, 28).to(device)

    saturation_1 = eval_one_sample(net, input_1)
    saturation_2 = eval_one_sample(net, input_2)

    saturation_1 = torch.hstack(saturation_1)
    saturation_2 = torch.hstack(saturation_2)
    
    assert torch.all(saturation_1 == saturation_2)
    if verbose:
        print("********************")
        print("Check saturations:", torch.all(saturation_1 == saturation_2).item())
        print("********************")


# ---------------------- #
# Optimization functions #
# ---------------------- #

# Scipy #

def compute_error_scipy(net, net_approx, comp_net, sample, output_dir, 
                        nb_constraints="all", start=0, end=1, device="cpu", verbose=False):
                
        net = copy.deepcopy(net)                # deep copy for safety reasons
        net_approx = copy.deepcopy(net_approx)  # deep copy for safety reasons
        comp_net = copy.deepcopy(comp_net)      # deep copy for safety reasons

        sample = sample.to(device).double()

        # Start timer
        start_time = time.time()
        
        # 2. Solve the NLP
        # (i) Compute coefficients of the NLP
        p = 0.7
        W, b, W_1, b_1 = objective_coeff(comp_net, sample, mode="np")
        A_reduced, bounds = constraints_coeff(comp_net, sample)
        if nb_constraints != "all":
            A_reduced = A_reduced[:nb_constraints]
            bounds = bounds[:nb_constraints]

        # (ii) Wrap the constraint in dict form for minimize()
        constraints = [
            {
            'type': 'ineq', 
            'fun': constraint_xi_0,
            'jac' : jac_constraint_xi_0, # provide analytic Jacobian
            'args': (W_1, b_1, p)
            },
            {
            'type': 'ineq', 
            'fun': constraints_xj_s, 
            'jac': jac_constraints_xj_s, # provide analytic Jacobian
            'args': (A_reduced, bounds)
            }
                    ]

        # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
        input_bounds = Bounds([-0.5]*W.shape[1], [2.9]*W.shape[1])

        # (iii) Initial guess: sample itself
        x0 = sample.flatten().cpu().numpy()

        if verbose:
            print("Checking contraints at x0:")
            xi_0 = constraint_xi_0(x0, W_1, b_1, p=0.7)
            print("Constraint xi_0 ≥ 0:\t", xi_0 >= 0)
            xi_js = constraints_xj_s(x0, A_reduced, bounds)
            print("Constraints xi_j's ≥ 0:\t", (xi_js >= 0).all())
        
        # (iv) Run minimization
        method = 'trust-constr' # 'trust-constr' (better but slower), 'SLSQP'

        options = {
            'maxiter': 10000,
            'disp': True,
            'sparse_jacobian' : True, # improves a lot!
            'xtol' : 1e-6,
            'gtol': 1e-6,           # Gradient norm tolerance
        }

        iteration = [0]

        def callback_fn(xk, state=None):
            iteration[0] += 1
            if verbose:
                print(".", end="", flush=True)

        res = minimize(
                    objective_fn_np, x0, args=(W, b, "cross-entropy"), # objective
                    jac=grad_fn_np,             # gradients: (better without??? not clear...)
                    bounds=input_bounds,        # bounds 
                    constraints=constraints,    # contraints
                    method=method,
                    # hess=lambda x, *args: np.zeros((len(x), len(x))),
                    options=options,
                    callback=callback_fn
                    )
        
        # Compute sample error (3 methods)
        objective_value, real_error, computed_error = compute_errors(net, net_approx, comp_net, sample, 
                                                                        W, b, loss_fn="cross-entropy")

        # results: error_1, error_2, error_3 (should coincide) and ERROR IN POLYTOPE
        with open(f"{output_dir}/results_{start}_{end}_nlp.csv", "a") as f:
            f.write(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-res.fun}\n")

        if verbose:
            
            # Checks
            print("🔍 Errors:", real_error, computed_error, objective_value)
            assert abs(real_error - computed_error) < TOL
            assert abs(computed_error - objective_value) < TOL
            assert abs(objective_value - real_error) < TOL
            print("✅ Errors' consistency check passed.\n")

            print("✅ Objective value:", res.fun)
            print("✅ Optimal solution:", res.x.shape)
            sol = x0.reshape(28, 28)        # XXX
            import matplotlib.pyplot as plt # XXX
            plt.imshow(sol)                 # XXX
            plt.show()                      # XXX
            sol = res.x.reshape(28, 28)     # XXX
            import matplotlib.pyplot as plt # XXX
            plt.imshow(sol)                 # XXX
            plt.show()                      # XXX
            xl, xu = -0.5, 2.9
            inside_bounds = np.all(x0 >= xl) and np.all(x0 <= xu)
            constraints_satisfied_1 = (constraint_xi_0(x0, W_1, b_1, p) >= 0)
            constraints_satisfied_2 = (constraints_xj_s(x0, A_reduced, bounds) >= 0).all()
            constraints_satisfied = constraints_satisfied_1 and constraints_satisfied_2
            print(f"🔍 x0 inside variable bounds:\t\t {inside_bounds}")
            print(f"🔍 Constraints satisfied at x0:\t\t {constraints_satisfied}")
            assert inside_bounds and constraints_satisfied , "❌ Constraints falied at x0!"
            print("✅ Constraints and bounds checks passed.")

            inside_bounds = np.all(res.x >= xl) and np.all(res.x <= xu)
            constraints_satisfied_1 = (constraint_xi_0(res.x, W_1, b_1, p) >= 0)
            constraints_satisfied_2 = (constraints_xj_s(res.x, A_reduced, bounds) >= 0).all()
            constraints_satisfied = constraints_satisfied_1 and constraints_satisfied_2
            print(f"🔍 Sol. inside variable bounds:\t\t {inside_bounds}")
            print(f"🔍 Constraints satisfied at sol:\t {constraints_satisfied}")
            assert inside_bounds and constraints_satisfied , "❌ Constraints falied at sol!"
            print("✅ Constraints and bounds checks passed.")

            # End timer
            end_time = time.time()
            elapsed_time = end_time - start_time    
            print(f"\nOptimization time {elapsed_time:.4f} seconds")
            
            print("\nErrors\n------")
            # check gradient
            grad_err = check_grad(objective_fn_np, grad_fn_np, x0, W, b, "cross-entropy")
            print("Gradient error:", grad_err)

            # check jacobians
            def wrapper_1(x):
                return constraint_xi_0(x, W_1, b_1, p=0.7)
            def wrapper_2(x):
                return constraints_xj_s(x, A_reduced, bounds)
            
            J_numeric_1 = approx_derivative(wrapper_1, x0)
            J_analytic_1 = jac_constraint_xi_0(x0, W_1, b_1, p=0.7)
            print("Jacobian #1 error:", np.max(np.abs(J_numeric_1 - J_analytic_1)))

            J_numeric_2 = approx_derivative(wrapper_2, x0)
            J_analytic_2 = jac_constraints_xj_s(x0, A_reduced, bounds)
            print("Jacobian #2 error:", np.max(np.abs(J_numeric_2 - J_analytic_2)))

            print("\nSanity checks\n-------------")


# IPOPT #


def compute_error_ipopt(net, net_approx, comp_net, sample, output_dir, 
                        nb_constraints="all", start=0, end=1, device="cpu", verbose=False):

    net = copy.deepcopy(net)                # deep copy for safety reasons
    net_approx = copy.deepcopy(net_approx)  # deep copy for safety reasons
    comp_net = copy.deepcopy(comp_net)      # deep copy for safety reasons

    sample = sample.to(device).double()

    # Start timer
    start_time = time.time()
    
    # 2. Solve the NLP
    # (i) Compute coefficients of the NLP
    p = 0.7
    W, b, W_1, b_1 = objective_coeff(comp_net, sample, mode="np")
    A_reduced, bounds = constraints_coeff(comp_net, sample)
    if nb_constraints != "all":
        A_reduced = A_reduced[:nb_constraints]
        bounds = bounds[:nb_constraints]

    # Initial guess: sample itself
    x0 = sample.flatten().cpu().numpy()

    m = A_reduced.shape[0] + 1
    n = A_reduced.shape[1]

    # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
    xl = np.ones(n, dtype=np.float64)*(-0.5)
    xu = np.ones(n, dtype=np.float64)*2.9

    # Constraints' bounds: [0, ∞)
    cl = np.concatenate([np.zeros(m - 1), [0.0]])
    cu = np.concatenate([np.full(m - 1, np.inf), [np.inf]])
    
    problem_obj = ToyProblem(W, b, W_1, b_1, A_reduced, bounds)

    nlp = cyipopt.Problem(
            n=n,    # nb of variables
            m=m,    # nb of constraints
            lb=xl,  # lower bounds
            ub=xu,  # upper bounds
            cl=cl,  # constraints lower bounds
            cu=cu,  # constraints upper bounds
            problem_obj=problem_obj
        )

    print_level = 5 if verbose==True else 1
    nlp.add_option("print_level", print_level)
    nlp.add_option("tol", 1e-6)
    nlp.add_option("hessian_approximation", "limited-memory") # XXX
    constr_tol = 1e-6
    nlp.add_option("constr_viol_tol", constr_tol)

    if verbose:

        # Checks
        check_shapes_consistency(A_reduced, x0, cl, cu, xl, xu, verbose)
        check_feasibility(problem_obj, x0, xl, xu, cl, cu, constr_tol, verbose)
        check_objective_gradient(problem_obj, x0, verbose=verbose)
        check_constraint_jacobian(problem_obj, x0, verbose=verbose)

    solution, info = nlp.solve(x0) # solve problem

    # Compute sample error (3 methods)
    objective_value, real_error, computed_error = compute_errors(net, net_approx, comp_net, sample, 
                                                                    W, b, loss_fn="cross-entropy")

    # results: error_1, error_2, error_3 (should coincide) and ERROR IN POLYTOPE
    with open(f"{output_dir}/results_{start}_{end}_nlp.csv", "a") as f:
        f.write(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-info["obj_val"]}\n")        

    if verbose:
        print("\nErrors at x0 (1,2,3) and maximal error (4)")
        print(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-info["obj_val"]}")

        print("\n✅ Optimal solution:", solution.shape)
        print("Objective value:", info["obj_val"])
        check_feasibility(problem_obj, solution, xl, xu, cl, cu, constr_tol)

        check_objective_value(solution, -info["obj_val"], 
                            net, net_approx, comp_net, 
                            W, b, loss_fn="cross-entropy", verbose=True)
        
        check_predictions_consistency(x0, comp_net)
        check_predictions_consistency(solution, comp_net)



if __name__ == "__main__":

    # Parameters
    config = parse_config()

    method = config["method"]
    print(f"Using  solver: {method}\n")
    verbose = config["verbose"]

    DEVICE = config.get("device", "cpu")
    print(f"Using device: {DEVICE}\n")

    model_name = config["model_name"]
    start = config["start"]
    end = config["end"]
    bits = config["bits"]
    output_dir = config["output_dir"]
        
    # Dataset
    test_dataset = create_dataset(mode="experiment")
    subset_dataset = Subset(test_dataset, list(range(start, end)))

    # Networks
    NETWORK = os.path.join("checkpoints", model_name)
    MODEL = SmallDenseNet 
    # LAYERS = 4
    # INPUT_SIZE = (1, 28, 28) 
    # N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK, device=DEVICE)
    net_approx = load_network(MODEL, NETWORK, device=DEVICE)
    comp_net = create_comparing_network(net, net_approx, bits=bits, skip_magic=True)

    # Compute errors
    for sample, _ in tqdm(subset_dataset, desc="Processing"):

        sample = sample.to(DEVICE).double()

        if method == "scipy":
            compute_error_scipy(net, net_approx, comp_net, sample, output_dir, 
                                nb_constraints="all", start=start, end=end, device=DEVICE, verbose=verbose)
        
        elif method == "iüpopt":
            compute_error_ipopt(net, net_approx, comp_net, sample, output_dir, 
                                nb_constraints="all", start=start, end=end, device=DEVICE, verbose=verbose)
