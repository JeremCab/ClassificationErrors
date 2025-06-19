import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from utils.network import SmallDenseNet

class Quantization():
    
    def __init__(self, net, bits=8):
   
        parameters = torch.hstack([p.flatten() for p in net.parameters()])
        self.min = parameters.min().item()
        self.max = parameters.max().item()

        
        self.a = - 2**(bits-1)
        self.b = 2**(bits-1) - 1

        self.s = (self.max - self.min) / (self.b - self.a)
        self.z = int((self.max * self.a - self.min * self.b) / (self.max - self.min))

        
    def quant_round(self, number):

        # convert down to "bits" bits (such as 8 bits)
        q_number = np.round(1 / self.s * number + self.z, decimals=0)
        q_number = np.clip(q_number, a_min=self.a, a_max=self.b)

        # convert back to float64
        q_number = q_number.astype(np.int64)
        new_number = self.s * (q_number - self.z)
        new_number = new_number.astype(np.float64)

        return new_number
    
    def convert(self, network):

        for p in network.parameters():
            print(p.shape)
            p_value = p.cpu().detach().numpy()
            new_p_value = self.quant_round(p_value)
            new_p = torch.Tensor(new_p_value).double() #, dtype=torch.float64)
            # p.copy_ requires grad, p.data is not very nice way :(
            # p.copy_(new_p)
            p.data = new_p

        
        return network

    
def lower_precision(net, bits=16):
    if bits == 16:
        return net.half().double()
    else:
        quant = Quantization(net, bits)
        return quant.convert(net).cuda()

    
if __name__ == "__main__":

    NETWORK =  "mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    
    net = MODEL()
    net.load_state_dict(torch.load(NETWORK))
    net.eval()
    net.double()

    quant = Quantization(net)

    print(quant.quant_round(0.87))
    print(quant.quant_round(0.12))

    input = torch.randn(1, 1, 28, 28, dtype=torch.float64)

    print(net(input))
    
    net = quant.convert(net)

    print(net(input))

    # ---------------------------------------
    net = MODEL()
    net.load_state_dict(torch.load(NETWORK))
    net.eval()
    net.double()
    quant = Quantization(net, bits=4)
    net = quant.convert(net)
    print(net(input))
