import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# def create_dataset(train=True, batch_size=8, num_workers=2):
#     dataset_name = "mnist"

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     dataset = MNIST(root="./data", train=train, download=True, transform=transform)

#     # dataset = CIFAR10(root='./data', train=train, download=True, transform=transform)


#     if train:
#         train_set, val_set = random_split(dataset, [50000, 10000])

#         train_loader = DataLoader(
#             train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
#         )

#         val_loader = DataLoader(
#             val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
#         )

#         return train_loader, val_loader, dataset_name

#     test_loader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
#     )
#     return test_loader



def create_dataset(batch_size=512, num_workers=0, val_split=0.1, data_root="./data", mode="train"):
    """
    Creates and returns train, validation, and test data loaders for MNIST.
    
    Args:
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading.
        val_split (float): Fraction of training data to use for validation.
        data_root (str): Directory to store/download MNIST data.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset_name)
    """
    generator = torch.Generator().manual_seed(42)

    dataset_name = "mnist"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full train and test datasets
    full_train_dataset = MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=data_root, train=False, download=True, transform=transform)

    # Split train dataset into train and validation
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if mode != "train":
        return test_dataset
    
    return train_loader, val_loader, test_loader, dataset_name
