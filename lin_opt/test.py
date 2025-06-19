import click
import numpy as np
import tqdm
import torch
import torch.nn as nn
import sys

from lin_opt.network_old import SmallDenseNet, SmallConvNet
from lin_opt.dataset_old import create_dataset
from quant_utils import Quantization

TOL = 1e-8
TOL2 = 1e-9

def eval_one_sample(net, sample):
    """evaluates one sample and returns boolean vector 
    of relu's saturations"""
    saturations = []
    
    outputs = sample
    if not isinstance(net, nn.Sequential):
        net = next(iter(net.children()))
    assert isinstance(net, nn.Sequential)
    
    for layer in net:
        outputs = layer(outputs)
        if isinstance(layer, nn.ReLU):
            saturations.append(outputs != 0)
            #saturations.append(torch.logical_not(outputs.isclose(torch.tensor(0, dtype=torch.float64), atol=TOL, rtol=0)))     
    return saturations

def eval_one_sample_test(net, sample):
    """evaluates one sample and returns boolean vector 
    of relu's saturations"""
    saturations = []
    
    outputs = sample
    assert isinstance(net, nn.Sequential)
    for layer in net:
        outputs = layer(outputs)
        if isinstance(layer, nn.ReLU):
            saturations.append(torch.logical_not(outputs.isclose(torch.tensor(0, dtype=torch.float64), atol=TOL, rtol=0)))     
    return saturations


def prune_network(net, saturations):
    layers = []
    assert isinstance(net, nn.Sequential)
    for layer in net:
        layers.append(layer)
            
    for j, saturation in enumerate(saturations):

        # find the j-th ReLU layer and get previous Linear layer
        jj = j 
        for i, l in enumerate(layers):
            if isinstance(l, nn.ReLU):
                if jj == 0:
                    break
                else:
                    jj -= 1
        i -= 1
        assert isinstance(layers[i], nn.Linear)
    
        W, b = layers[i].weight, layers[i].bias
        saturation = saturation.flatten()

        # filter out previous linear layer
        W2 = W[saturation]
        b2 = b[saturation]

        new_pre_layer = nn.Linear(W2.shape[1], W2.shape[0]).double()
        new_pre_layer.weight.data = W2
        new_pre_layer.bias.data = b2

        layers[i] = new_pre_layer

        # find next Linear layer 
        i += 1
        while not isinstance(layers[i], nn.Linear): 
            i += 1
        assert isinstance(layers[i], nn.Linear)
        W, b = layers[i].weight, layers[i].bias

        W2 = W[:, saturation]
        b2 = torch.clone(b)

        new_post_layer = nn.Linear(W2.shape[1], W2.shape[0]).double()
        new_post_layer.weight.data = W2
        new_post_layer.bias.data = b2
        # bias stays the same

        layers[i] = new_post_layer
    
    # create a fixed network
    net = nn.Sequential(*layers).cuda().eval()
    
    print(net)
    return net

def squeeze_network(net):
    layers = []
    assert isinstance(net, nn.Sequential)
    for layer in net:
        layers.append(layer)

    # get rid of ReLU (network already pruned)
    layers = [l for l in layers if not isinstance(l, nn.ReLU)]
    # we  do not need dropout (only eval mode)
    layers = [l for l in layers if not isinstance(l, nn.Dropout)]

    # check that all layers are linear (first can be flatten)
    assert isinstance(layers[0], nn.Flatten) or isinstance(layers[0], nn.Linear)
    for l in layers[1:]:
        assert isinstance(l, nn.Linear)
    

    # take only linear layers 
    lin_layers = [l for l in layers if isinstance(l, nn.Linear)] 

    W = [l.weight.data for l in lin_layers[::-1]]
    b = [l.bias.data for l in lin_layers[::-1]]

    
    W_new = torch.linalg.multi_dot(W)
    bias_new = b[0]
    for i, bias in enumerate(b[1:]):
        ii = i + 1
        if ii > 1:
            W_inter = torch.linalg.multi_dot(W[:ii])
        else:
            W_inter = W[0]
        bias_new += torch.mm(W_inter, bias.reshape((-1, 1))).flatten()

        
    new_layer = nn.Linear(W[-1].shape[1], W[0].shape[0]).double()
    new_layer.weight.data = W_new
    new_layer.bias.data = bias_new

    new_layers = [new_layer]
    if isinstance(layers[0], nn.Flatten):
        new_layers = [layers[0]] + new_layers

    return nn.Sequential(*new_layers).cuda().eval()
    
    
def load_network(network_class, network_path):
    net = network_class()
    net.load_state_dict(torch.load(network_path))
    net.eval()
    net.cuda()
    net.double()
    return  net


def test_squeeze():
    
    BATCH_SIZE=1    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)
    net = next(iter(net.children())) # extract the nn.Sequential

    data = create_dataset(train=False, batch_size=BATCH_SIZE)

    for inputs, labels in data:
        inputs = inputs.cuda().double()
        assert inputs.shape[0] == 1 # one sample in a batch
        saturations = eval_one_sample(net, inputs)
        
        pnet = prune_network(net, saturations)

        outputs = net(inputs)
        outputs2 = pnet(inputs)
        print(nn.functional.mse_loss(outputs, outputs2))

        snet = squeeze_network(pnet)
        print(snet)

        outputs3 = snet(inputs)
        print(outputs)
        print(outputs3)
        print(nn.functional.mse_loss(outputs, outputs3))

        assert nn.functional.mse_loss(outputs, outputs3).isclose(torch.tensor(0.0, dtype=torch.float64))
        print("oh yes")

def test_compnet():

    BATCH_SIZE=1    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)
    compnet = create_comparing_network(net, net2)

    net2 = lower_precision(net2)
    
    print(compnet)

    data = create_dataset(train=False, batch_size=BATCH_SIZE)

        
    for inputs, labels in data:
        inputs = inputs.cuda().double()

        output = net(inputs)
        output2 = net2(inputs)

        err1 = (output-output2).abs().sum()
        err2 = compnet(inputs)

        assert err1.isclose(err2)
        print("oh yes")

def test_squeezed_compnet():

    BATCH_SIZE=1    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)
    compnet = create_comparing_network(net, net2)

    net2 = lower_precision(net2)
    
    print(compnet)

    data = create_dataset(train=False, batch_size=BATCH_SIZE)

        
    for inputs, labels in data:
        inputs = inputs.cuda().double()

        output = net(inputs)
        output2 = net2(inputs)

        err1 = (output-output2).abs().sum()
        err2 = compnet(inputs)

        assert err1.isclose(err2)

        saturations = eval_one_sample(compnet, inputs)        
        target_net = squeeze_network(prune_network(compnet, saturations))

        err3 = target_net(inputs)
        assert err1.isclose(err3)
        
        print("oh yes")

        
def lower_precision(net, bits=16):
    if bits == 16:
        return net.half().double()
    else:
        quant = Quantization(net, bits)
        return quant.convert(net).cuda()

def stack_linear_layers(layer1, layer2, common_input=False):

    if common_input:
        wide_W1 =  layer1.weight.data
        wide_W2 =  layer2.weight.data 
    else:
        wide_W1 = torch.hstack([layer1.weight.data,
                                torch.zeros(*layer1.weight.data.shape).double().cuda()])
        wide_W2 = torch.hstack([torch.zeros(*layer2.weight.data.shape).double().cuda(),
                                layer2.weight.data])
                            
    new_weight = torch.vstack([wide_W1, wide_W2])

    new_layer = nn.Linear(new_weight.shape[1], new_weight.shape[0]).double()
    new_layer.weight.data = new_weight
    new_layer.bias.data = torch.hstack([layer1.bias.data, layer2.bias.data])
    
    return new_layer

def magic_layer(layer1, layer2):
    """ equation (13) and (14) in Jirka's document """
    
    W1 = layer1.weight.data
    b1 = layer1.bias.data

    W2 = layer2.weight.data
    b2 = layer2.bias.data

    # magic_b1 = b1 - b2
    # mabic_b2 = b2 - b1
    magic_b = torch.hstack([b1-b2, b2-b1])

    # magic W  =  W1 -W2
    #            -W1  W2 
    magic_W = torch.vstack(
        [
            torch.hstack([W1, -W2]),
            torch.hstack([-W1, W2])
        ]
    )

    new_layer = nn.Linear(magic_W.shape[1], magic_W.shape[0]).double()
    new_layer.weight.data = magic_W
    new_layer.bias.data = magic_b

    return new_layer
    
    
    
def create_comparing_network(net, net2, bits=16):
    twin = lower_precision(net2, bits=bits) 

    layer_list = []

    sequence1 = next(iter(net.children()))
    assert isinstance(sequence1, nn.Sequential)

    sequence2 = next(iter(twin.children()))
    assert isinstance(sequence2, nn.Sequential)

    first_linear = True
    
    for layer1, layer2  in zip(sequence1, sequence2):
        if isinstance(layer1, nn.Flatten):
            assert isinstance(layer2, nn.Flatten)
            layer_list.append(layer1)
        elif isinstance(layer1, nn.Dropout):
            assert isinstance(layer2, nn.Dropout)
            layer_list.append(layer1)
        elif isinstance(layer1, nn.ReLU):
            assert isinstance(layer2, nn.ReLU)
            layer_list.append(layer1)
        elif isinstance(layer1, nn.Linear):
            assert isinstance(layer2, nn.Linear)
            layer_list.append(stack_linear_layers(layer1, layer2, common_input=first_linear))
            first_linear = False
        else:
            raise NotImplementedError

    assert isinstance(sequence1[-1], nn.Linear)
    assert isinstance(sequence2[-1], nn.Linear)
    assert isinstance(layer_list[-1], nn.Linear)

    layer_list = layer_list[:-1]

    layer_list.append(magic_layer(sequence1[-1], sequence2[-1]))
    
        
    layer_list.append(nn.ReLU())
    
    output_layer = nn.Linear(20, 1).double() # TODO fix  the number
    output_layer.weight.data = torch.ones(1, 20).double()
    output_layer.bias.data = torch.zeros(1).double()

    layer_list.append(output_layer)
    
    return nn.Sequential(*layer_list).cuda()

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

def get_subnetwork(net, i):
    # network up to i-th linear layer
    layers = []
    for layer in net:
        layers.append(layer)
        if isinstance(layer, nn.Linear):
            i -= 1
        if i < 0:
            break
    return nn.Sequential(*layers)
    

def create_upper_bounds(net, inputs):

    # extract the sequential 
    #    net = next(iter(net.children())) NO NEED FOR COMPNET
    assert isinstance(net, nn.Sequential)
    
    saturations = eval_one_sample(net, inputs)

    A_list = []
    bound_list = [] 
    for i, saturation in enumerate(saturations):
        subnet = get_subnetwork(net, i)
        if i == 0:
            target = subnet
        else:
            target = squeeze_network(prune_network(subnet, saturations[:i]))

        W = target[-1].weight.data
        b = target[-1].bias.data

        # saturation: True ~ U, False ~ S   
        W_lower = W[torch.logical_not(saturation).flatten()]
        b_lower = b[torch.logical_not(saturation).flatten()].reshape(-1, 1)
        W_higher = W[saturation.flatten()]
        b_higher = b[saturation.flatten()].reshape(-1, 1)

        bound_for_lower = torch.full((W_lower.shape[0],), -TOL, dtype=torch.float64)
        bound_for_higher = torch.full((W_higher.shape[0],), -TOL, dtype=torch.float64)
        
        W = torch.vstack([W_lower, -1*W_higher])
        b = torch.vstack([b_lower, -1*b_higher])
        
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

    assert res.success
    
    return res.fun, res.x

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
    
    input2 = torch.tensor(input2).reshape(1, 1, 28, 28).cuda()

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
    
    BATCH_SIZE=1    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)
    compnet = create_comparing_network(net, net2, bits=bits)
    
    print(compnet)

    data = create_dataset(train=False, batch_size=BATCH_SIZE)

    i = 0    
    for i, (inputs, labels) in enumerate(data):
        if i < start or i >= end:
            continue
        inputs = inputs.cuda().double()

        out1 = net(inputs)
        out2 = net2(inputs)
        real_error = (out2 - out1).abs().sum().item()
        computed_error = compnet(inputs).item()
        
        
        # min c @ x
        c = -1*create_c(compnet, inputs)

        # A_ub @ x <= b_ub
        A_ub, b_ub = create_upper_bounds(compnet, inputs)
        #b_ub = torch.zeros((A_ub.shape[0],), dtype=torch.float64)
        #b_ub = torch.full((A_ub.shape[0],), -TOL, dtype=torch.float64)
        
        # A_eq @ x == b_eq
        A_eq = torch.zeros((1, N+1)).double()
        A_eq[0, 0] = 1.0
        b_eq = torch.zeros((1,)).double()
        b_eq[0] = 1.0                    

        # l <= x <= u 
        l = -0.5
        u = 3.0

        err, x = optimize(c, A_ub, b_ub, A_eq, b_eq, l, u) 
        print("result:", -err)

        assert np.isclose(x[0], 1.0)

        y = torch.tensor(x[1:], dtype=torch.float64).reshape(1, -1).cuda()
        err_by_net = compnet(y).item()
        
        err_by_sol = (c @ torch.tensor(x, dtype=torch.float64).cuda()).item()

        try: 
            assert np.isclose(-err, err_by_net)
            assert np.isclose(err, err_by_sol)
            
            check_upper_bounds(A_ub, b_ub, inputs, x[1:])
            check_saturations(net, inputs, x[1:])
        except AssertionError:
            print(" *** Optimisation FAILED. *** ")
            continue
        
        with open(f"{outputdir}/results_{start}_{end}.csv", "a") as f:
            print(f"{real_error:.6f},{computed_error:.6f},{-err:.6f}", file=f)
        #np.save(f"{RESULT_PATH}/{i}.npy", np.array(x[1:], dtype=np.float64))
        #np.save(f"{RESULT_PATH}/{i}_orig.npy", inputs.cpu().numpy())
        i += 1
        
if __name__ == "__main__":

    # test_squeeze() # 1.
    #test_compnet() # 2.
    #test_squeezed_compnet() # 3.


    main()
