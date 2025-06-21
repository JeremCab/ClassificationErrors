import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm, trange

from config import DEVICE
from utils.dataset import create_dataset
from utils.network import SimpleNet, DenseNet, SmallDenseNet, SmallConvNet


def train(net, train_data, val_data, optimizer, criterion, num_epochs, device):
    for epoch in trange(num_epochs, desc="Epochs"):

        # --- Training ---
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_data:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        # --- Validation ---
        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_data:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = 100 * val_correct / val_total

        tqdm.write(f"Epoch {epoch+1:02d} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    train_data, val_data, test_data, dataset_name = create_dataset(batch_size=args.batch_size)

    net = SmallDenseNet().to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0)

    train(net, train_data, val_data, optimizer, criterion, 
          num_epochs=args.num_epochs, device=DEVICE)
    
    model_name = net.__class__.__name__.lower()  # set model name

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        args.checkpoint_dir, f"{dataset_name}_{model_name}_{args.num_epochs}.pt"
    )

    torch.save(net.state_dict(), checkpoint_path)
    print(f"Training complete. Network saved: {checkpoint_path}")
