import sys
import os
import json

import traceback
import concurrent.futures

from tqdm import tqdm
YELLOW = "\033[93m"
RESET = "\033[0m"

import argparse
import yaml

import copy

import numpy as np
import cyipopt
import nlopt

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

# Don't need this line since "export PYTHONPATH=$(pwd)" in train_network.sh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from propagate_intervals.train import parse_config
from preprocessing import LossHead, create_comparing_network, eval_one_sample
from utils.network import load_network, SmallConvNet, VerySmallDenseNet, SmallDenseNet
from utils.dataset import create_dataset, select_confident_subdataset
from linear_utils import create_c, create_upper_bounds, optimize
from nonlinear_utils import *
from linear_utils import TOL, TOL2


def check_upper_bounds(A, b, input_1, input_2, verbose=False):
    """
    This function checks the inequalities constraints A @ x <= ε of the linear program (LP)
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

# *** Scipy *** #

def compute_error_scipy(net, net_approx, comp_net, sample, p=0.7,
                        filename="results.csv", 
                        nb_constraints="all", 
                        loss_fn="cross-entropy", 
                        device="cpu", 
                        tol=1e-6, 
                        verbose=False):
                
        net = copy.deepcopy(net)                # deep copy for safety reasons
        net_approx = copy.deepcopy(net_approx)  # deep copy for safety reasons
        comp_net = copy.deepcopy(comp_net)      # deep copy for safety reasons

        sample = sample.to(device).double()

        # Start timer
        start_time = time.time()
        
        # Non-Linear Problem (NLP)
        # Coefficients
        W, b, W_1, b_1 = objective_coeff(comp_net, sample, mode="np")
        A_reduced, bounds = constraints_coeff(comp_net, sample)
        if nb_constraints != "all":
            A_reduced = A_reduced[:nb_constraints]
            bounds = bounds[:nb_constraints]
        
        m = A_reduced.shape[0] + 1
        n = A_reduced.shape[1]

        # Bounds
        # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
        xl = np.ones(n, dtype=np.float64)*(-0.5)
        xu = np.ones(n, dtype=np.float64)*2.9
        # Constraints' bounds: [0, ∞)
        cl = np.concatenate([np.zeros(m - 1), [0.0]])
        cu = np.concatenate([np.full(m - 1, np.inf), [np.inf]])

        # Constraint in dict form for minimize()
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

        # Bounds
        # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
        input_bounds = Bounds([-0.5]*W.shape[1], [2.9]*W.shape[1])

        # Initial guess: sample itself
        x0 = sample.flatten().cpu().numpy()

        # Run minimization
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
                    objective_fn_np, x0, args=(W, b, loss_fn), # objective
                    jac=grad_fn_np,             # gradients: (better without??? not clear...)
                    bounds=input_bounds,        # bounds 
                    constraints=constraints,    # constraints
                    method=method,
                    # hess=lambda x, *args: np.zeros((len(x), len(x))),
                    options=options,
                    callback=callback_fn
                    )
        
        # Compute sample error (3 methods)
        objective_value, real_error, computed_error = compute_errors(net, net_approx, comp_net, sample, 
                                                                     W, b, loss_fn=loss_fn)

        # results: error_1, error_2, error_3 (should coincide) and polytope error
        with open(filename, "a") as f:
            f.write(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-res.fun}\n")

        if verbose:
            
            print("\n-----------------------\n")

            check_shapes_consistency(A_reduced, x0, cl, cu, xl, xu, verbose)
            check_bounds_and_constraints(constraints, x0, xl, xu, cl, cu, constr_tol=1e-6, verbose=verbose)

            print("\n-----------------------")


            print("\nErrors at x0 (1,2,3) and maximal error (4)")
            print(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-res.fun}")

            print("\n✅ Optimal solution:", res.x.shape)
            print("Objective value:", res.fun)
            check_bounds_and_constraints(constraints, res.x, xl, xu, cl, cu, constr_tol=1e-6, verbose=verbose)

            check_objective_value(res.x, -res.fun, 
                                net, net_approx, comp_net, 
                                W, b, loss_fn=loss_fn, verbose=verbose)
            
            check_predictions_consistency(x0, comp_net)
            check_predictions_consistency(res.x, comp_net)

            end_time = time.time()
            elapsed_time = end_time - start_time    
            print(f"\nOptimization time: {elapsed_time:.4f} seconds")

            print("\n-----------------------\n")
            
            # check gradient
            grad_err = check_grad(objective_fn_np, grad_fn_np, x0, W, b, loss_fn)
            print("🔍 Gradient error:", grad_err)

            # check jacobians
            def wrapper_1(x):
                return constraint_xi_0(x, W_1, b_1, p)
            def wrapper_2(x):
                return constraints_xj_s(x, A_reduced, bounds)
            
            J_numeric_1 = approx_derivative(wrapper_1, x0)
            J_analytic_1 = jac_constraint_xi_0(x0, W_1, b_1, p)
            print("🔍 Jacobian #1 error:", np.max(np.abs(J_numeric_1 - J_analytic_1)))

            J_numeric_2 = approx_derivative(wrapper_2, x0)
            J_analytic_2 = jac_constraints_xj_s(x0, A_reduced, bounds)
            print("🔍 Jacobian #2 error:", np.max(np.abs(J_numeric_2 - J_analytic_2)))

        print("\n-----------------------\n")


# *** IPOPT *** #


def compute_error_ipopt(net, net_approx, comp_net, sample, p=0.7,
                        filename="results.csv", 
                        nb_constraints="all", 
                        loss_fn="cross-entropy", 
                        device="cpu", 
                        tol=1e-6, 
                        verbose=False):

    net = copy.deepcopy(net)                # deep copy for safety reasons
    net_approx = copy.deepcopy(net_approx)  # deep copy for safety reasons
    comp_net = copy.deepcopy(comp_net)      # deep copy for safety reasons

    sample = sample.to(device).double()

    # Start timer
    start_time = time.time()
    
    # # Non-Linear Problem (NLP)
    # Ccoefficients
    W, b, W_1, b_1 = objective_coeff(comp_net, sample, mode="np")
    A_reduced, bounds = constraints_coeff(comp_net, sample)
    if nb_constraints != "all":
        A_reduced = A_reduced[:nb_constraints]
        bounds = bounds[:nb_constraints]

    # Initial guess: sample itself
    x0 = sample.flatten().cpu().numpy()

    m = A_reduced.shape[0] + 1
    n = A_reduced.shape[1]

    # Bounds
    # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
    xl = np.ones(n, dtype=np.float64)*(-0.5)
    xu = np.ones(n, dtype=np.float64)*2.9

    # Constraints' bounds: [0, ∞)
    cl = np.concatenate([np.zeros(m - 1), [0.0]])
    cu = np.concatenate([np.full(m - 1, np.inf), [np.inf]])
    
    problem_obj = NonLinearProblem(W, b, W_1, b_1, A_reduced, bounds, p=p)

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
    nlp.add_option("tol", tol)
    nlp.add_option("hessian_approximation", "limited-memory")
    constr_tol = tol
    nlp.add_option("constr_viol_tol", constr_tol)

    # Run minimization
    solution, info = nlp.solve(x0) # solve problem

    # Compute sample error (3 methods)
    objective_value, real_error, computed_error = compute_errors(net, net_approx, comp_net, sample, 
                                                                 W, b, loss_fn=loss_fn)

    # results: error_1, error_2, error_3 (should coincide) and polytope error
    with open(filename, "a") as f:
        f.write(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-info["obj_val"]}\n")        

    if verbose:

        print("\n-----------------------\n")

        check_shapes_consistency(A_reduced, x0, cl, cu, xl, xu, verbose)
        check_bounds_and_constraints(problem_obj, x0, xl, xu, cl, cu, constr_tol=constr_tol, verbose=verbose)
        check_objective_gradient(problem_obj, x0, loss_fn=loss_fn, verbose=verbose)
        check_constraint_jacobian(problem_obj, x0, verbose)
        check_predictions_consistency(x0, comp_net)

        print("\n-----------------------")

        print("\nErrors at x0 (1,2,3) and maximal error (4)")
        print(f"{real_error:.8f},{computed_error:.8f},{objective_value:.8f},{-info["obj_val"]}")
        print("\n✅ Optimal solution:", solution.shape)
        print("Objective value:", info["obj_val"])
        check_bounds_and_constraints(problem_obj, solution, xl, xu, cl, cu, constr_tol=constr_tol, verbose=verbose)
        check_objective_value(solution, -info["obj_val"], 
                            net, net_approx, comp_net, 
                            W, b, loss_fn=loss_fn, verbose=verbose)
        check_predictions_consistency(solution, comp_net)

        print("\n-----------------------\n")


# *** NLopt *** #


def compute_error_nlopt(net, net_approx, comp_net, sample, p=0.7, 
                        filename="results.csv",
                        nb_constraints="all", 
                        loss_fn="cross-entropy", 
                        device="cpu", 
                        nb_iter=15000, 
                        tol=1e-6, 
                        verbose=False):
            
    net = copy.deepcopy(net)                # deep copy for safety reasons
    net_approx = copy.deepcopy(net_approx)  # deep copy for safety reasons
    comp_net = copy.deepcopy(comp_net)      # deep copy for safety reasons

    sample = sample.to(device).double()

    # Start timer
    start_time = time.time()
    
    # Non-Linear Problem (NLP)
    # Coefficients
    W, b, W_1, b_1 = objective_coeff(comp_net, sample, mode="np")
    A_reduced, bounds = constraints_coeff(comp_net, sample)
    if nb_constraints != "all":
        A_reduced = A_reduced[:nb_constraints]
        bounds = bounds[:nb_constraints]
    
    m = A_reduced.shape[0] + 1
    n = A_reduced.shape[1]
    
    # Bounds
    # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
    xl = np.ones(n, dtype=np.float64)*(-0.5)
    xu = np.ones(n, dtype=np.float64)*2.9
    # Constraints' bounds: [0, ∞)
    cl = np.concatenate([np.zeros(m - 1), [0.0]])
    cu = np.concatenate([np.full(m - 1, np.inf), [np.inf]])

    # Problem
    opt = nlopt.opt(nlopt.AUGLAG, n)
    opt.set_maxeval(nb_iter)
    opt.set_ftol_rel(tol)
    opt.set_xtol_rel(tol)

    local_opt = nlopt.opt(nlopt.LD_MMA, n)
    local_opt.set_maxeval(nb_iter)  # Try small inner loop budget
    local_opt.set_ftol_rel(tol)
    local_opt.set_xtol_rel(tol)
    opt.set_local_optimizer(local_opt)

    # Safe lower and upper bounds after dataset transformation: [-0.5, 2.9]
    opt.set_lower_bounds([-0.5]*n)
    opt.set_upper_bounds([2.9]*n)

    obj_fn = objective_fn_nlopt(W, b, loss_fn=loss_fn, verbose=verbose)
    opt.set_min_objective(obj_fn)   # minimize!

    # Constraints (linear and non-linear)
    # A_minus x + b_minus ≤ 0       <=>   
    # -A_reduced x - bounds ≤ 0     <=>
    # -(-W) x - (-b) ≤ 0            <=>
    # W x + b ≤ 0                   (eq. (7)-(12) √)
    A_minus = -A_reduced
    b_minus = -bounds

    def linear_constraints_vectorized(result, x, grad):
        if grad.size > 0:
            grad[:] = A_minus
        result[:] = A_minus @ x + b_minus
        return None # nlopt requirement

    opt.add_inequality_mconstraint(linear_constraints_vectorized, [tol] * A_minus.shape[0])
    opt.add_inequality_constraint(lambda x, grad: constraint_xi_0_nlopt(x, grad, W_1, b_1, p=p), tol)

    # Initial guess
    x0 = sample.flatten().cpu().numpy()
    x_opt = opt.optimize(x0)
    dummy_grad = np.zeros_like(x_opt)
    obj_val = objective_fn_nlopt(W, b, loss_fn=loss_fn, verbose=verbose)(x_opt, dummy_grad)

    # Compute predictions
    prob_1, class_1, _, prob_2, class_2, _ = get_predictions(x0, comp_net)
    prob_opt_1, class_opt_1, logits_opt_1, prob_opt_2, class_opt_2, logits_opt_2 = get_predictions(x_opt, comp_net)

    # Compute sample error (3 methods)
    # results: error_1, error_2, error_3 (should coincide) and polytope error
    objective_value, real_error, computed_error = compute_errors(net, net_approx, comp_net, sample, 
                                                                 W, b, loss_fn=loss_fn) # XXX takes time, no need 3 methods!!!
    errors = real_error, computed_error, objective_value

    # Write results
    header = (
        "error_x0_a,"
        "error_x0_b,"
        "error_x0_c,"
        "error_polytope,"
        "net_prob_x0,"
        "net_class_x0,"
        "net_approx_prob_x0,"
        "net_approx_class_x0,"
        "net_prob_x_opt,"
        "net_class_x_opt,"
        "net_logits_x_opt,"
        "net_approx_prob_x_opt,"
        "net_approx_class_x_opt,"
        "net_approx_logits_x_opt\n"
    )

    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0

    with open(filename, "a") as f:
        if write_header:
            f.write(header)
        f.write(
            f"{errors[0]:.8f},"
            f"{errors[1]:.8f},"
            f"{errors[2]:.8f},"
            f"{-obj_val:.8f},"
            f"{prob_1:.8f},{class_1},"
            f"{prob_2:.8f},{class_2},"
            f"{prob_opt_1:.8f},{class_opt_1},"
            f"\"{json.dumps(logits_opt_1.tolist())}\","
            f"{prob_opt_2:.8f},{class_opt_2},"
            f"\"{json.dumps(logits_opt_2.tolist())}\"\n"
        )

    # Checks x0
    check_shapes_consistency(A_reduced, x0, cl, cu, xl, xu, verbose)
    check_bounds_and_constraints(opt, x0, xl, xu, cl, cu, 
                                 W_1=W_1, b_1=b_1, A_reduced=A_reduced, bounds=bounds, 
                                 p=p, constr_tol=tol, verbose=verbose)
    check_objective_gradient(opt, x0, W=W, b=b, verbose=verbose)
    # check_constraint_jacobian(problem_obj, x0, verbose)
    # check_predictions_consistency(x0, comp_net, verbose=verbose)      # equal predictions of N and Ñ not required
    
    # Checks x_opt
    check_objective_value(x_opt, -obj_val, 
                    net, net_approx, comp_net, 
                    W, b, loss_fn=loss_fn, verbose=verbose)
    check_bounds_and_constraints(opt, x_opt, xl, xu, cl, cu, 
                                 W_1=W_1, b_1=b_1, A_reduced=A_reduced, bounds=bounds, 
                                 p=p, constr_tol=tol, verbose=verbose)
    # check_predictions_consistency(x_opt, comp_net, verbose=verbose)   # equal predictions of N and Ñ not required


# *** Worker function *** #


def process_sample_batch(args):
    batch_idx, batch, config = args
    results = []
    for i, (sample, _) in batch:
        sample = sample.to(config["device"]).double()
        try:
            if config["method"] == "scipy":
                res = compute_error_scipy(
                    config["net"], 
                    config["net_approx"], 
                    config["comp_net"], 
                    sample, 
                    config["p"],
                    filename = config["filename"], 
                    nb_constraints="all", 
                    loss_fn="cross-entropy", 
                    device=config["device"],
                    tol=config["tol"], 
                    verbose=config["verbose"]
                )

            elif config["method"] == "ipopt":
                res = compute_error_ipopt(
                    config["net"], 
                    config["net_approx"], 
                    config["comp_net"], 
                    sample, 
                    config["p"],
                    filename = config["filename"], 
                    nb_constraints="all", 
                    loss_fn="cross-entropy", 
                    device=config["device"],
                    tol=config["tol"], 
                    verbose=config["verbose"]
                )

            elif config["method"] == "nlopt":
                res = compute_error_nlopt(
                    config["net"], 
                    config["net_approx"], 
                    config["comp_net"], 
                    sample, 
                    config["p"],
                    filename = config["filename"], 
                    nb_constraints="all", 
                    loss_fn="cross-entropy", 
                    device=config["device"],
                    nb_iter=config["nb_iter"], 
                    tol=config["tol"],
                    verbose=config["verbose"]
                )
            results.append(res)

        except Exception:
            print(f"❌ Exception occurred for sample {i}...\n")
            traceback.print_exc()

    return results  # return results for this batch


def chunked_iterable(iterable, batch_size):
    """Yield successive chunks from iterable."""
    it = iter(iterable)
    while True:
        batch = list()
        try:
            for _ in range(batch_size):
                batch.append(next(it))
        except StopIteration:
            if batch:
                yield batch
            break
        yield batch


# =================== #
# *** Main script *** #
# =================== #


if __name__ == "__main__":

    # Parameters
    MODEL_CLASSES = {
        "SmallDenseNet": SmallDenseNet,
        "VerySmallDenseNet": VerySmallDenseNet
    }
        
    config = parse_config()
    config["tol"] = float(config["tol"])
    config["device"] = config.get("device", "cpu")
    config["multiprocessing"] = config.get("multiprocessing", False)
    config["batch_size"] = config.get("batch_size", 8)  # NEW: batch size for workers

    print(f"Using solver: {config['method']}")
    print(f"Using device: {config['device']}")
    print(f"Multiprocessing: {config['multiprocessing']}")
    print(f"Batch size: {config['batch_size']}")
    
    MODEL = MODEL_CLASSES[config["model_class"]]
    NETWORK = os.path.join("checkpoints", config["model_name"])

    net = load_network(MODEL, NETWORK, device=config["device"])
    net_copy = copy.deepcopy(net)

    test_subset = config['test_subset']
    test_dataset = create_dataset(mode="experiment")

    if test_subset is not None:
        mask = torch.isin(test_dataset.targets, torch.tensor(test_subset))
        indices = torch.nonzero(mask, as_tuple=True)[0]
        test_dataset = Subset(test_dataset, indices)
    print(f"test subset:", test_subset)

    test_dataset = select_confident_subdataset(
        net_copy, test_dataset,
        p_threshold=config["p"], 
        batch_size=512, 
        device=config["device"]
    )

    if config["end"] is not None:
        subset_dataset = Subset(test_dataset, list(range(config["start"], config["end"])))
    else:
        subset_dataset = test_dataset

    print(f"""
Filtering test set: keeping test samples that are correctly classified\
by the original network with prob p ≥ {config['p']}"""
)
    print(f"=> {len(test_dataset)} remaining samples.\n")

    net_approx = load_network(MODEL, NETWORK, device=config["device"])
    comp_net = create_comparing_network(net, net_approx, bits=config["bits"], skip_magic=True)

    config["net"] = net
    config["net_approx"] = net_approx
    config["comp_net"] = comp_net

    def none2all(param):
        return "all" if param is None else param
    start = config['start']
    end_ = none2all(config['end'])
    test_subset_ = none2all(config['test_subset'])
    p = config['p']
    filename = f"results_{start}_{end_}_testsubset_{test_subset_}_p={p}_nlp.csv"
    config["filename"] = os.path.join(config['output_dir'], filename)

    # --- Optimization ---
    if config["multiprocessing"]:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            batches = [
                (batch_idx, batch, config)
                for batch_idx, batch in enumerate(
                    chunked_iterable(enumerate(subset_dataset), config["batch_size"])
                )
            ]
            list(tqdm(
                executor.map(process_sample_batch, batches),
                total=len(batches),
                desc="Processing...",
                colour="green"
            ))
    else:
        # Sequential fallback (process one by one)
        for i, (sample, _) in enumerate(tqdm(subset_dataset, desc="Processing...", colour="green")):
            process_sample_batch((i, [(i, (sample, _))], config))
            


# √ load correct network
# √ debug code
# √ implement dataset selection
# √ implement saving x0 predictions
# √ fix start-end thing (add all option)...
# csv column names
# implement iconip results (notebook)