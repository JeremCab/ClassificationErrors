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


def check_upper_bounds(A, b, input1, input2):

    A = A.cpu()

    print(A.shape)
    
    input1 = input1.cpu()
    
    input1 = torch.hstack([torch.tensor(1),
                           input1.reshape(-1)])
    input2 = torch.hstack([torch.tensor(1),
                           torch.tensor(input2)])
    print(input1.shape)
    print(input2.shape)

    result = A @ input1
    print("Check upper bounds 1: ", torch.all(result <= TOL + TOL2))
    assert torch.all(result <= TOL + TOL2)
    
    result = A @ input2 
    print("Check upper bounds 2: ", torch.all(result <= TOL + TOL2 ))
    assert torch.all(result <= TOL + TOL2)
    
    wrong_indexes = torch.logical_not(result <= TOL + TOL2)
    print(wrong_indexes.sum())
    
    print(result[wrong_indexes])

    

def check_saturations(net, input1, input2):
    
    device = next(net.parameters()).device

    input2 = torch.tensor(input2).reshape(1, 1, 28, 28).to(device)

    saturation1 = eval_one_sample(net, input1)
    saturation2 = eval_one_sample(net, input2)

    saturation1 = torch.hstack(saturation1)
    saturation2 = torch.hstack(saturation2)
    
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
    
    print("*** Networks N and Ñ side by side\n\n", compnet)

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
        
        # XXX Additional equality constraints (not in the paper)
        # to account for the constant part of the LP, i.e., y_0 = 1
        # A_eq @ x == b_eq 
        A_eq = torch.zeros((1, N+1)).double()
        A_eq[0, 0] = 1.0
        b_eq = torch.zeros((1,)).double()
        b_eq[0] = 1.0                    

        # l <= x <= u 
        l = -0.5
        u = 3.0

        # err, x = optimize(c, A_ub, b_ub, A_eq, b_eq, l, u) 
        # print("result:", -err)
        res = optimize(c, A_ub, b_ub, A_eq, b_eq, l, u) # XXX MY FIX
        err = res.fun # XXX MY FIX
        x = res.x     # XXX MY FIX

        assert np.isclose(x[0], 1.0)

        # y is the solution, i.e., the input with maximal error
        y = torch.tensor(x[1:], dtype=torch.float64).reshape(1, -1).to(DEVICE)
        err_by_net = compnet(y).item() # gives the error related to this solution
        
        # sanity check that the network really procudes the same error as the one computed by the LP
        err_by_sol = (c @ torch.tensor(x, dtype=torch.float64).to(DEVICE)).item()

        try: 
            assert np.isclose(-err, err_by_net) # sanity check!!!
            assert np.isclose(err, err_by_sol)  # sanity check!!!
            
            check_upper_bounds(A_ub, b_ub, inputs, x[1:]) # check that the solution is correct!!!
            check_saturations(net, inputs, x[1:])         # check that the solution is in the correct saturation region!!!
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
