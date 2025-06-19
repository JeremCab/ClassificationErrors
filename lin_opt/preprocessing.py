import numpy as np
import torch
import torch.nn as nn

from quant_utils import lower_precision

NUMCLASSES = 10 # fix this

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
    return saturations



def prune_network(net, saturations):
    """Creates a new network that is equivalent with the given net on the assumption that saturations
    are fixed. Unused neurons are deleted.
    """

    device = next(net.parameters()).device

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
    net = nn.Sequential(*layers).to(device).eval()
    
    print(net)
    return net

def squeeze_network(net):
    """ Assumes the net is already pruned for some particular saturation.
    Creates an equivalent network that has only one layer. (Squeeze the linear layers into one).
    This means computing the "shortcut weights" associated with the fixed saturation.
    """

    device = next(net.parameters()).device
    
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

    W = [l.weight.data for l in lin_layers[::-1]] # weights of reversed list of layers
    b = [l.bias.data for l in lin_layers[::-1]]   # biases of reversed list of layers

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

    return nn.Sequential(*new_layers).to(device).eval()
    

        

def stack_linear_layers(layer1, layer2, common_input=False):
    """ Stack two linear layers horizontally side by side. If common_input is True they share the same 
    input vector, otherwise the two inputs are processed separately.
    """
    device = next(layer1.parameters()).device

    if common_input:
        wide_W1 =  layer1.weight.data
        wide_W2 =  layer2.weight.data 
    else:
        wide_W1 = torch.hstack([layer1.weight.data,
                                torch.zeros(*layer1.weight.data.shape).double().to(device)])
        wide_W2 = torch.hstack([torch.zeros(*layer2.weight.data.shape).double().to(device),
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
    
    
    
def create_comparing_network(net, net2, bits=16, skip_magic=False):
    """Takes two networks and creates a super network, the two original networks are side by side and
    on the top they are connected to compute the sum of differences between their outputs. The
    second network is rounded for the given number of bits.

    """

    device = next(net.parameters()).device

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

    if not skip_magic:
        layer_list = layer_list[:-1]

        layer_list.append(magic_layer(sequence1[-1], sequence2[-1]))
    
        
        layer_list.append(nn.ReLU())
    
        output_layer = nn.Linear(20, 1).double() # TODO fix  the number
        output_layer.weight.data = torch.ones(1, 20).double()
        output_layer.bias.data = torch.zeros(1).double()
  
        layer_list.append(output_layer)
    
    return nn.Sequential(*layer_list).to(device)

def get_subnetwork(net, i):
    """ Returns network up to i-th linear layer includisively."""

    layers = []
    for layer in net:
        layers.append(layer)
        if isinstance(layer, nn.Linear):
            i -= 1
        if i < 0:
            break
    return nn.Sequential(*layers)
    

def create_comparing_network_classifier(net, label, other, in_orig=False):

    device = next(net.parameters()).device

    out = net[-1].out_features
    assert out == 2 * NUMCLASSES 

    if not in_orig:
        n = out // 2
    else:
        n = 0 
    
    output_layer = nn.Linear(out, 1).double() # TODO fix the number
    output_layer.weight.data = torch.zeros(1, out).double()
    output_layer.bias.data = torch.zeros(1).double()
    output_layer.weight.data[0, n + label] = -1.0
    output_layer.weight.data[0, n + other] = 1.0

    old_layers = [layer for layer in net]
    old_layers.append(output_layer)

    return nn.Sequential(*old_layers).to(device)


