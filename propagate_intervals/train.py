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

        running_loss = 0.0
        correct = 0
        num = 0
        i = 0
        net.train()
        for data in train_data:
            i += 1
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            num += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #        if i % 10 == 9:    # print every 2000 mini-batches
    #    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f} acc: {100*correct/num:.3f}')

        with torch.no_grad():
            net.eval()
            num, correct = 0, 0
            running_loss = 0
            i = 0
            for inputs, labels in val_data:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels)        
                _, predicted = torch.max(outputs.data, 1)
                num += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                i += 1
        tqdm.write(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f} vall acc: {100*correct/num:.3f}')

        running_loss = 0.0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    train_data, val_data, dataset_name = create_dataset(train=True, batch_size=args.batch_size)

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
    print(f"Training complete. Network saved to {checkpoint_path}.")
