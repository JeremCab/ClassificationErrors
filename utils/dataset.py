from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def create_dataset(train=True, batch_size=8, num_workers=2):
    dataset_name = "mnist"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNIST(root="./data", train=train, download=True, transform=transform)

    # dataset = CIFAR10(root='./data', train=train, download=True, transform=transform)

    if train:
        train_set, val_set = random_split(dataset, [50000, 10000])

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader, dataset_name

    test_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_loader
