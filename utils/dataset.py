import torch
import torchvision
import torchvision.transforms as transforms


CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def create_dataset(train=True, batch_size=8):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transforms.Normalize( (0.1307,), (0.3081,))]
    )


    
    dataset = torchvision.datasets.MNIST(root="./data", train=train,
                                         download=True, transform=transform)

#    print(len(dataset))
    
#    dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
#                                            download=True, transform=transform)
    if train:
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                                num_workers=2)
        return trainloader, valloader
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=2)

