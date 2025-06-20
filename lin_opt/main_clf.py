import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import click
import numpy as np
import torch

from preprocessing import create_comparing_network, eval_one_sample
from preprocessing import create_comparing_network_classifier

from utils.network import load_network, SmallConvNet, SmallDenseNet
from utils.dataset import create_dataset
from linear_utils import create_c, create_upper_bounds, optimize
from linear_utils import TOL, TOL2


def check_upper_bounds(A, b, input1, input2):

    A = A.cpu()

    if input1 is not None:
        input1 = input1.cpu()
        input1 = torch.hstack([torch.tensor(1),
                               input1.reshape(-1)])

    if input2 is not None:
        input2 = torch.hstack([torch.tensor(1),
                               torch.tensor(input2)])

    if input1 is not None:
        result = A @ input1
        print("Check upper bounds 1: ", torch.all(result <= TOL + TOL2))
        if not torch.all(result <= TOL + TOL2):
            print("upper bounds 1 failed")
    
        assert torch.all(result <= TOL + TOL2)

    if input2 is not None:
        result = A @ input2 
        print("Check upper bounds 2: ", torch.all(result <= TOL + TOL2 ))
        if not torch.all(result <= TOL + TOL2):
            print("upper bounds 2 failed")
        assert torch.all(result <= TOL + TOL2)
    
        #   wrong_indexes = torch.logical_not(result <= TOL + TOL2)
        #   print(wrong_indexes.sum())
    
        #   print(result[wrong_indexes])


def check_saturations(net, input1, input2):
    
    device = next(net.parameters()).device
    
    input2 = torch.tensor(input2).reshape(1, 1, 28, 28).to(device)

    saturation1 = eval_one_sample(net, input1)
    saturation2 = eval_one_sample(net, input2)

    saturation1 = torch.hstack(saturation1)
    saturation2 = torch.hstack(saturation2)
    
    print("Check saturations", torch.all(saturation1 == saturation2).item())
    assert torch.all(saturation1 == saturation2)


@click.command()
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option("-b", "--bits", default=16)
@click.option("--outputdir", default="results")

def main(start, end, bits, outputdir): 
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE=1    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK, device=DEVICE)
    net2 = load_network(MODEL, NETWORK, device=DEVICE)
    compnet = create_comparing_network(net, net2, bits=bits, skip_magic=True)
    
    print(compnet)
    
    data = create_dataset(train=False, batch_size=BATCH_SIZE)

    for i, (inputs, labels) in enumerate(data):
        print(f"***************** Computation {i} *****************") # XXX
        maximas = []
        label = labels[0].item()

        if i < start or i >= end:
            continue

        inputs = inputs.to(DEVICE).double()

        outputs = net(inputs)
        _, preds = outputs.max(1)

        if preds[0].item() != label:
            print("wrong prediction, skipping")
            continue

        for other in range(10): # TODO FIX number 10 to number classes
            if other == label:
                continue
            print(f"other {other}", flush=True)
            
            """check ... works
            out1 = net(inputs)
            out2 = net2(inputs)
            real_error = (out2 - out1).abs().sum().item()
            computed_error = compnet(inputs)#.item()

            print(real_error)
            diff = (computed_error[0, :10] - computed_error[0, 10:]).abs().sum()
        

            print(diff)
            """ 

            # comparing_network_classifier computes net2(other) - net(label)
            # it shold be maximised to get wrong prediction
            # XXX HAVEN'T WE ALREADY CREATED THE NETWORKS compnet ABOVE?
            compnet_classifier = create_comparing_network_classifier(compnet, label, other)
            
            """
            # second check
            out2 = net2(inputs)
            print(out2[0, other] - out2[0, label])
            co = compnet(inputs)[0, 10:]
            print(co[other] - co[label])

            cco = compnet_classifier(inputs)
            print(cco)
            """
            
            # min c @ x
            c = -1 * create_c(compnet_classifier, inputs)

            # compnet classifier is destroyed after this step, why?

            """check
            x_input = torch.tensor([1.0, *inputs.flatten()], dtype=torch.float64).to(device)
            
            print(c @ x_input)
            """
            
            # A_ub @ x <= b_ub
            A_ub, b_ub = create_upper_bounds(compnet, inputs)

            print("First check")
            check_upper_bounds(A_ub, b_ub, inputs, None)
            
            # add conditions on net(label) > net(others)
            for other_output in range(10):
                if other_output == label:
                    continue
                cf = create_comparing_network_classifier(compnet, label, other_output, in_orig=True)
                c_cf = create_c(cf, inputs).reshape(1, -1)

                A_ub = torch.cat([A_ub, c_cf])
                b_ub = torch.cat([b_ub, torch.tensor([-TOL], dtype=torch.float64)])

            print("Check net(label) > net(others)")
            check_upper_bounds(A_ub, b_ub, inputs, None)

                
            # add condition net2(label) < net2(other)
            cf = create_comparing_network_classifier(compnet, label, other, in_orig=False)
            c_cf = -1*create_c(cf, inputs).reshape(1, -1)

            A_ub = torch.cat([A_ub, c_cf])
            b_ub = torch.cat([b_ub, torch.tensor([-TOL], dtype=torch.float64)])
            
            
            # A_eq @ x == b_eq
            A_eq = torch.zeros((1, N+1)).double()
            A_eq[0, 0] = 1.0
            b_eq = torch.zeros((1,)).double()
            b_eq[0] = 1.0                    

            # l <= x <= u 
            l = -0.5
            u = 3.0

            res = optimize(c, A_ub, b_ub, A_eq, b_eq, l, u)
            if res.success:
                
                err, x = res.fun, res.x
                print("result:", -err)

                assert np.isclose(x[0], 1.0)
            
                y = torch.tensor(x[1:], dtype=torch.float64).reshape(1, -1).to(DEVICE)
                err_by_net = compnet_classifier(y).item()
                
                err_by_sol = (c @ torch.tensor(x, dtype=torch.float64).to(DEVICE)).item()
                
                try: 
                    check_upper_bounds(A_ub, b_ub, None, x[1:])
                    check_saturations(net, inputs, x[1:])

                    if not np.isclose(-err, err_by_net):
                        print(-err, err_by_net, "failed -err and err_by_net")
                        assert False

                    if not np.isclose(err, err_by_sol):
                        print(err, err_by_sol, "failed err and err_by_sol")
                        assert False
            
                except AssertionError:
                    print(" *** Optimisation FAILED. *** ")
                    maximas.append(None)
                    continue
                maximas.append(-err )

            else:
                print("Solution not found.")
                maximas.append(None)
        
        with open(f"{outputdir}/results_{start}_{end}.csv", "a") as f:
            values = [val for val in maximas if val is not None]
            if values:
                max_value = f"{max(values):.6f}"
            else:
                max_value = "NaN"
            print(f"{i},{max_value},{maximas}", file=f)
        
if __name__ == "__main__":

    #test_squeeze() # 1.
    #test_compnet() # 2.
    #test_squeezed_compnet() # 3.


    main()
