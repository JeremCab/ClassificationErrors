import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import click
import numpy as np
import torch

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


def check_saturations(net, input1, input2, verbose=False):
    
    device = next(net.parameters()).device

    input2 = torch.tensor(input2).reshape(1, 1, 28, 28).to(device)

    saturation1 = eval_one_sample(net, input1)
    saturation2 = eval_one_sample(net, input2)

    saturation1 = torch.hstack(saturation1)
    saturation2 = torch.hstack(saturation2)
    
    if verbose:
        print("Check saturations", torch.all(saturation1 == saturation2).item())
    
    assert torch.all(saturation1 == saturation2)



def compute_errors_lp(model_name, start, end, bits, outputdir): 
    
    BATCH_SIZE=1    
    NETWORK=f"checkpoints/{model_name}"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK, device=DEVICE)
    net2 = load_network(MODEL, NETWORK, device=DEVICE)
    compnet = create_comparing_network(net, net2, bits=bits)
    
    data = create_dataset(train=False, batch_size=BATCH_SIZE)

    i = 0    
    for i, (inputs, labels) in enumerate(data):
        if i < start or i >= end:
            continue
        inputs = inputs.to(DEVICE).double()

        out1 = net(inputs)
        out2 = net2(inputs)
        real_error = (out2 - out1).abs().sum().item()
        computed_error = compnet(inputs).item()
        
        # Objective function
        # min c @ x
        c = -1*create_c(compnet, inputs)

        # Inequality constraints
        # A_ub @ x <= b_ub
        A_ub, b_ub = create_upper_bounds(compnet, inputs)
        #b_ub = torch.zeros((A_ub.shape[0],), dtype=torch.float64)
        #b_ub = torch.full((A_ub.shape[0],), -TOL, dtype=torch.float64)
        
        # Additional equality constraints (not in the paper)
        # to account for the constant part of the LP, i.e., y_0 = 1
        # A_eq @ x == b_eq 
        A_eq = torch.zeros((1, N+1)).double()
        A_eq[0, 0] = 1.0
        b_eq = torch.zeros((1,)).double()
        b_eq[0] = 1.0                    

        # l <= x <= u 
        l = -0.5
        u = 3.0

        res = optimize(c, A_ub, b_ub, A_eq, b_eq, l, u)
        err = res.fun
        x = res.x

        assert np.isclose(x[0], 1.0)

        # y is the solution, i.e., the input with maximal error
        y = torch.tensor(x[1:], dtype=torch.float64).reshape(1, -1).to(DEVICE)
        err_by_net = compnet(y).item() # gives the error related to this solution
        
        # sanity check that the network really procudes the same error as the one computed by the LP
        err_by_sol = (c @ torch.tensor(x, dtype=torch.float64).to(DEVICE)).item()

        try: 
            assert np.isclose(-err, err_by_net) # sanity check!
            assert np.isclose(err, err_by_sol)  # sanity check!
            
            # check inequality constraints for the data sample (inputs) and the sol of the LP (x[1:])
            check_upper_bounds(A_ub, b_ub, inputs, x[1:])
            check_saturations(net, inputs, x[1:])         # check that the solution is in the correct saturation region!
        except AssertionError:
            print(" *** Optimisation FAILED. *** ")
            # pass
        
        with open(f"{outputdir}/results_{start}_{end}.csv", "a") as f:
            print(f"{real_error:.6f},{computed_error:.6f},{-err:.6f}", file=f)
        #np.save(f"{RESULT_PATH}/{i}.npy", np.array(x[1:], dtype=np.float64))
        #np.save(f"{RESULT_PATH}/{i}_orig.npy", inputs.cpu().numpy())
        i += 1
        
if __name__ == "__main__":

    # test_squeeze() # 1.
    #test_compnet() # 2.
    #test_squeezed_compnet() # 3.

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
