import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# Don't need this line since "export PYTHONPATH=$(pwd)" in test_network.sh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import create_dataset
from utils.network import SimpleNet, DenseNet, VerySmallDenseNet, SmallDenseNet, SmallConvNet
from propagate_intervals.train import parse_config
from optimization.quant_utils import lower_precision


def evaluate(net, test_loader, criterion, device):
    net.to(device)
    net.eval()   # set to evaluation mode
    
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Batch processing:", colour="green"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / total
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")
    return avg_loss, accuracy


if __name__ == "__main__":

    # Parameters
    config = parse_config()

    preferred_device = config.get("device", "cpu")

    if preferred_device == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    batch_size = config["batch_size"]
    checkpoint_dir = config["checkpoint_dir"]
    checkpoint_model = config["checkpoint_model"]
    bits = config["bits"]
    test_subset = config["test_subset"]

    print(f"Checkpoint model: {checkpoint_dir}/{checkpoint_model}")
    print(f"Using device: {DEVICE}\n")

    # Test set
    test_dataset = create_dataset(mode="experiment")

    if test_subset is not None:
        mask = torch.isin(test_dataset.targets, torch.tensor(test_subset))
        indices = torch.nonzero(mask, as_tuple=True)[0]
        test_dataset = Subset(test_dataset, indices)
    print(f"test subset:", test_subset)

    
    print(f"Length of the test set: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model_name = checkpoint_model.split("_")[1]

    if model_name == "verysmalldensenet":
        net = VerySmallDenseNet().to(DEVICE)
    elif model_name == "smalldensenet":
        net = SmallDenseNet().to(DEVICE)
    elif model_name == "densenet":
        net = DenseNet().to(DEVICE)

    net.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_model)))
    criterion = nn.CrossEntropyLoss()
    print("\nTesting original network...\n", net)
    
    # Testing net 
    test_loss_1, test_acc_1 = evaluate(net, test_loader, criterion, DEVICE)

    # Testing net approx
    net_approx = lower_precision(net, bits=bits)
    print(f"\nTesting {bits}-bit rounded network...\n", net_approx)
    test_loss_2, test_acc_2 = evaluate(net_approx, test_loader, criterion, DEVICE)

    output_dir = config["output_dir"]
    model_name = Path(checkpoint_model).stem
    with open(f"{output_dir}/{model_name}_test_results.csv", "w") as f:
        f.write(f"{test_acc_1},{test_acc_2}\n")
    print(f"Results saved: {output_dir}/{model_name}_test_results.csv\n")


# √ Create a yaml file and a bash script...
# √ Improve the script such that it can evaluate a subset of the dataset!!!
# √ Improve the script such that it can evaluate the rounded version of the network!!!
# saving results...
