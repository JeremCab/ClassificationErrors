import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from utils.network import SmallDenseNet

# class Quantization():
    
#     def __init__(self, net, bits=8):
   
#         parameters = torch.hstack([p.flatten() for p in net.parameters()])
#         self.min = parameters.min().item()
#         self.max = parameters.max().item()

#         self.a = - 2**(bits-1)
#         self.b = 2**(bits-1) - 1

#         self.s = (self.max - self.min) / (self.b - self.a)
#         self.z = int((self.max * self.a - self.min * self.b) / (self.max - self.min))

        
#     def quant_round(self, number):

#         # convert down to "bits" bits (such as 8 bits)
#         q_number = np.round(1 / self.s * number + self.z, decimals=0)
#         q_number = np.clip(q_number, a_min=self.a, a_max=self.b)

#         # convert back to float64
#         q_number = q_number.astype(np.int64)
#         new_number = self.s * (q_number - self.z)
#         new_number = new_number.astype(np.float64)

#         return new_number
    
#     def convert(self, network):

#         for p in network.parameters():
#             p_value = p.cpu().detach().numpy()
#             new_p_value = self.quant_round(p_value)
#             new_p = torch.Tensor(new_p_value).double() #, dtype=torch.float64)
#             # p.copy_ requires grad, p.data is not very nice way :(
#             # p.copy_(new_p)
#             p.data = new_p
        
#         return network

# UPDATED CLASS
class Quantization:
    def __init__(self, net: torch.nn.Module, bits=8, device=None):
        # Set device; default to the device of the network
        if device is None:
            device = next(net.parameters()).device
        self.device = device

        # Pick dtype: float32 for MPS, float64 otherwise
        self.dtype = torch.float32 if str(device).startswith("mps") else torch.float64

        # Flatten all parameters to compute min/max
        parameters = torch.cat([p.flatten() for p in net.parameters()]).to(self.dtype)
        self.min = parameters.min().item()
        self.max = parameters.max().item()

        # Quantization range
        self.a = -2**(bits-1)
        self.b = 2**(bits-1) - 1

        # Scale and zero-point
        self.s = (self.max - self.min) / (self.b - self.a)
        self.z = int((self.max * self.a - self.min * self.b) / (self.max - self.min))

    def quant_round(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantizes and dequantizes a tensor using the specified bit-width.
        Chooses float32 for MPS, float64 otherwise.
        """
        # Ensure correct dtype
        tensor = tensor.to(self.dtype)

        # Quantize
        q_tensor = torch.round(tensor / self.s + self.z)
        q_tensor = torch.clamp(q_tensor, min=self.a, max=self.b)

        # Dequantize
        new_tensor = self.s * (q_tensor - self.z)
        return new_tensor.to(self.dtype)

    def convert(self, network: torch.nn.Module) -> torch.nn.Module:
        """
        Quantizes all parameters of the network in-place and moves them to the target device.
        """
        for p in network.parameters():
            new_p = self.quant_round(p).to(self.device)
            p.data = new_p

        return network

    
# def lower_precision(net, bits=16):

#     device = next(net.parameters()).device

#     if bits == 16:
#         return net.half().double()
#     else:
#         quant = Quantization(net, bits)
#         return quant.convert(net).to(device)    # .cuda()


# UPDATED FUNCTION
def lower_precision(net, bits=16):
    device = next(net.parameters()).device

    # For bits <= 16, just use Quantization (device-aware)
    quant = Quantization(net, bits, device=device)
    return quant.convert(net)


    
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
