import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallDenseNetOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 2000),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
#            nn.Linear(256, 128),
#            nn.ReLU(),
            nn.Linear(1000, 10)
        )
    def forward(self, x):
        return self.network(x)


class SmallDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
#            nn.Linear(256, 128),
#            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.network(x)
    

class SmallConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Flatten(),
            nn.Dropout(0.8),
            nn.Linear(12544, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
        
    def forward(self, xb):
        return self.network(xb)



    
class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 32000),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(32000, 16000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16000, 8000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8000, 16000),
            nn.ReLU(),
            nn.Linear(16000, 10)
#            nn.Dropout(0.2),
#            nn.Linear(8000, 2000),
#            nn.ReLU(),
#            nn.Dropout(0.1),
#            nn.Linear(2000, 512),
#            nn.ReLU(),
#            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        return self.network(x)
    

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Dropout(0.7),
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
        
    def forward(self, xb):
        return self.network(xb)

def load_network(network_class, network_path, device):
    net = network_class()
    net.load_state_dict(torch.load(network_path, weights_only=True))
    net.eval()
    net.to(device)
    net.double()
    return  net
