import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import argparse
import numpy as np

import torch
from torch.utils.data import Subset

from config import DEVICE
from preprocessing import create_comparing_network, eval_one_sample
from utils.network import load_network, SmallConvNet, SmallDenseNet
from utils.dataset import create_dataset
from linear_utils import create_c, create_upper_bounds, optimize
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



def compute_errors_lp(model_name, start, end, bits, outputdir): 
    
    NETWORK=f"checkpoints/{model_name}"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK, device=DEVICE)
    net_approx = load_network(MODEL, NETWORK, device=DEVICE)
    compnet = create_comparing_network(net, net_approx, bits=bits)
    
    test_dataset = create_dataset(mode="experiment")
    subset_dataset = Subset(test_dataset, list(range(start, end)))

    for sample, _ in tqdm(subset_dataset, desc="Processing"):

        sample = sample.to(DEVICE).double()

        out_1 = net(sample)
        out_2 = net_approx(sample)
        real_error = (out_2 - out_1).abs().sum().item()
        computed_error = compnet(sample).item()
        
        # Objective function
        # min c @ x
        c = -1*create_c(compnet, sample)

        # Inequality constraints
        # A_ub @ x <= b_ub
        A_ub, b_ub = create_upper_bounds(compnet, sample)
        #b_ub = torch.zeros((A_ub.shape[0],), dtype=torch.float64)
        #b_ub = torch.full((A_ub.shape[0],), -TOL, dtype=torch.float64)
        
        # Equality constraints (not in the paper) capturing the constant part of the LP, i.e., y_0 = 1
        # A_eq @ x == b_eq 
        A_eq = torch.zeros((1, N+1)).double()
        A_eq[0, 0] = 1.0
        b_eq = torch.zeros((1,)).double()
        b_eq[0] = 1.0                    

        # l <= x <= u 
        l = -0.5
        u = 3.0

        res = optimize(c, A_ub, b_ub, A_eq, b_eq, l, u)
        err = res.fun # Error computed by the LP
        x = res.x     # Solution of the LP

        assert np.isclose(x[0], 1.0)

        # y is the LP solution, i.e., the input in the saturation polytope yielding a maximal error
        y = torch.tensor(x[1:], dtype=torch.float64).reshape(1, -1).to(DEVICE)
        err_by_net = compnet(y).item()                                            # Error computed by compnet, i.e., |N(x) - Ñ(x)|
        err_by_sol = (c @ torch.tensor(x, dtype=torch.float64).to(DEVICE)).item() # Error computed by the LP (should be err)

        try: 
            assert np.isclose(-err, err_by_net) # sanity check
            assert np.isclose(err, err_by_sol)  # sanity check
            
            # Check inequality constraints for both the data sample and the LP sol
            check_upper_bounds(A_ub, b_ub, sample, x[1:])
            # Check that the data sample and the associated LP sol yield same saturations
            check_saturations(net, sample, x[1:])
        except AssertionError:
            print("Optimisation FAILED!")
            # pass
        
        with open(f"{outputdir}/results_{start}_{end}.csv", "a") as f:
            print(f"{real_error:.6f},{computed_error:.6f},{-err:.6f}", file=f)
        #np.save(f"{RESULT_PATH}/{i}.npy", np.array(x[1:], dtype=np.float64))
        #np.save(f"{RESULT_PATH}/{i}_orig.npy", inputs.cpu().numpy())


if __name__ == "__main__":

    # test_squeeze() # 1.
    # test_compnet() # 2.
    # test_squeezed_compnet() # 3.

    parser = argparse.ArgumentParser(description="Compute error bounds for a quantized network.")
    parser.add_argument("--model_name", type=str, help="Base model name")
    parser.add_argument("--start", type=int, default=0, help="Start index of the dataset")
    parser.add_argument("--end", type=int, default=10, help="End index of the dataset")
    parser.add_argument("--bits", type=int, default=16, help="Number of quantization bits")
    parser.add_argument("--outputdir", type=str, default="results", help="Output directory for results")

    args = parser.parse_args()

    compute_errors_lp(
        model_name=args.model_name,
        start=args.start,
        end=args.end,
        bits=args.bits,
        outputdir=args.outputdir
    )
