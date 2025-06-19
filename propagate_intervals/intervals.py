import tqdm
import torch
import torch.nn as nn
from torch.nn.functional import relu as R 

from network import SmallDenseNet, SmallConvNet
from propagate_intervals.dataset_old import create_dataset
from torch_conv_layer_to_fully_connected import torch_conv_layer_to_affine

RANDOM_DATO = None
ZERO = torch.tensor(0).double()

def mac(w, x):
    # torch.sum(w*x, dim=1)
    result = torch.zeros(w.shape[0]).double()
    for i in range(w.shape[1]):
        result += w[:,i]*x[i]
    return result

def negative(w): 
    return torch.min(ZERO, w)

def positive(w):
    return torch.max(ZERO, w)

def compute_input_ai_bi(class_=None):
    test_data = create_dataset(train=False, batch_size=BATCH_SIZE)

    a, b = [], []

    for x, label  in tqdm.tqdm(test_data):
        if class_ is not None:
            if label[0] != class_: # use class_ only with BATCH_SIZE==1
                continue
        x = x.double() 
        x = x.reshape(-1, N)
        amin, _ = torch.min(x, dim=0)
        bmax, _  = torch.max(x, dim=0)
        a.append(amin)
        b.append(bmax)
        
    a, _ = torch.min(torch.vstack(a), dim=0)
    b, _ = torch.max(torch.vstack(b), dim=0)

    return a, b

def compute_output_ai_bi(a, b, w, bias, relu=True):
    
    w_neg = negative(w)
    w_pos = positive(w)
      
    a_new = bias + mac(w_neg, b) + mac(w_pos, a)
    b_new = bias + mac(w_neg, a) + mac(w_pos, b)

    if relu:
        return R(a_new), R(b_new)
    else:
        return a_new, b_new

def compute_bias_shift(a, b, w, bias):
    bias = bias + mac(w, a)
    a, b = a-a, b-a
    return bias, a, b 

def load_network():
    net = MODEL()
    net.load_state_dict(torch.load(NETWORK))
    net.eval()
    return  net.double()

def get_layer(net, layer):

    x = torch.zeros(INPUT_SIZE).unsqueeze(0).double()
    input_size = INPUT_SIZE
    layer_pointer = None
    
    for child in net.children():
        assert isinstance(child, nn.Sequential)
        i = 0 # count only conv, fc and pool 
        for l in child.children():
            print(l.__class__.__name__)
            if isinstance(l, nn.Dropout):
                continue
            print("evaluating")
            x = l(x)
            if isinstance(l, nn.Flatten):
                continue 
            if isinstance(l, nn.ReLU):
                continue
            if i == layer:
                layer_pointer = l
                break
            input_size = x.shape 
            i += 1        

    if type(layer_pointer) == nn.Conv2d:
        fc = torch_conv_layer_to_affine(layer_pointer, input_size[-2:])
        return "conv", fc.weight, fc.bias, input_size, layer_pointer

    if type(layer_pointer) == nn.Linear:
        return "lin", layer_pointer.weight, layer_pointer.bias, input_size, layer_pointer

    if type(layer_pointer) == nn.MaxPool2d:
        return "pool", None, None, input_size, layer_pointer

    raise ValueError(f"{layer} is weird")


def delta(x):

    x_hat = x.half()
    x_hat = x_hat.double()

    return x_hat - x 

def hat(x):
    return x.half().double() 

def compute_output_alpha_beta(alpha, beta, b, delta_weight, delta_bias,
                              weight, bias, relu=True):
    weight_hat = hat(weight)
    bias_hat = hat(bias) 
    
    delta_neg = negative(delta_weight)
    delta_pos = positive(delta_weight)

    alpha_new = mac(delta_neg, b) #torch.sum(delta_neg*b, dim=1)
    beta_new = mac(delta_pos, b)  #torch.sum(delta_pos*b, dim=1)

    weight_hat_neg = negative(weight_hat)
    weight_hat_pos = positive(weight_hat)

    alpha_new += mac(weight_hat_pos, alpha) # torch.sum(weight_hat_pos*alpha, dim=1)
    alpha_new += mac(weight_hat_neg, beta) #torch.sum(weight_hat_neg*beta, dim=1)

    beta_new += mac(weight_hat_neg, alpha) #torch.sum(weight_hat_neg*alpha, dim=1)
    beta_new += mac(weight_hat_pos, beta) #torch.sum(weight_hat_pos*beta, dim=1)

    if relu:
        return (
            torch.minimum(ZERO, alpha_new + delta_bias),
            R(beta_new + delta_bias)
        )
    else:
        return alpha_new, beta_new
    
def compute_pool_alpha_beta(alpha, beta, shape, max_):
    alpha_new = alpha.reshape(*shape)
    beta_new = beta.reshape(*shape)

    return -1*max_(-1*alpha_new).flatten(), max_(beta_new).flatten()
    
    
def compute_pool_ai_bi(a, b, shape, max_):
    a_new = a.reshape(*shape)
    b_new = b.reshape(*shape)

    return max_(a_new).flatten(), max_(b_new).flatten()

def save(x, layer, name):
    torch.save(x, f"{NETWORK[:-3]}.{layer}.{name}.pt")
    print(f"{layer}: {name} saved")


def calculate_output_intervals(class_): 
    with torch.no_grad():
        a, b = compute_input_ai_bi(class_)
        alpha = torch.zeros(N).double()
        beta = torch.zeros(N).double()  
        net = load_network()

        for layer in range(LAYERS):
        
            type_, weight, bias, input_shape, layer_pointer = get_layer(net, layer)

            if type_ == "pool":
                alpha, beta = compute_pool_alpha_beta(alpha, beta, input_shape, layer_pointer)
                a, b = compute_pool_ai_bi(a, b, input_shape, layer_pointer)

            else:    
                bias, a, b = compute_bias_shift(a, b, weight, bias) 
                
                delta_weight, delta_bias = delta(weight), delta(bias)
                
                alpha, beta = compute_output_alpha_beta(alpha, beta, b, delta_weight, delta_bias,
                                                        weight, bias,
                                                        relu=layer!=(LAYERS-1)) 
            
                a, b = compute_output_ai_bi(a, b, weight, bias, relu=layer!=(LAYERS-1))
        
                del weight
                del bias
            assert torch.all(a<=b)
            assert torch.all(alpha<=0)
            assert torch.all(beta>=0)
                
            save(a, layer, "a")
            save(b, layer, "b")
            save(alpha, layer, "alpha")
            save(beta, layer, "beta")
            
        print("-----------------------")
        print(a)
        print(b)

        print(alpha)
        print(beta)
        print("ok")
    
    
    
if __name__ == "__main__":


    CLASS=0 
    BATCH_SIZE=1

    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 3 
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    calculate_output_intervals(CLASS) 

    """
    NETWORK="mnist_conv_net.pt"
    MODEL = SmallConvNet
    LAYERS = 5
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    calculate_output_intervals(CLASS) 
    """
