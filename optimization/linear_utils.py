import torch
import torch.nn as nn

from preprocessing import eval_one_sample, squeeze_network, prune_network, get_subnetwork

TOL = 1e-8   # almost 0: to check the bounds
TOL2 = 1e-9  # almost 0: to check the bounds

def create_c(compnet, inputs):
    assert inputs.shape[0] == 1 # one sample in a batch

    #    wide_inputs = torch.hstack([inputs, inputs]) TODO: delete this line

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
    # XXX We compute the shortcut weights of each layer j (squeeze the subnet up to layer j) => W_ji and W_j0
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


def optimize(c, A_ub, b_ub, A_eq, b_eq, l, u):
    c = c.cpu().numpy()
    A_ub, b_ub = A_ub.cpu().numpy(), b_ub.cpu().numpy()
    A_eq, b_eq = A_eq.cpu().numpy(), b_eq.cpu().numpy()
    
    from scipy.optimize import linprog

    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(l, u))
    print(res)
    
    return res
