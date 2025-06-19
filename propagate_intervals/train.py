import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn as nn

import tqdm

from utils.dataset import create_dataset
from utils.network import SimpleNet, DenseNet, SmallDenseNet, SmallConvNet


def train(net, train_data, val_data, optimizer, criterion, device, num_epochs):

    for epoch in tqdm.tqdm(range(num_epochs)):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        num = 0
        i = 0
        net.train()
        for data in tqdm.tqdm(train_data):
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
            for inputs, labels in tqdm.tqdm(val_data):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels)        
                _, predicted = torch.max(outputs.data, 1)
                num += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                i += 1
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f} vall acc: {100*correct/num:.3f}')

        running_loss = 0.0

        # if epoch % 50 == 0:
        #     end = input("End? yes/no: ")
        #     if end  == "yes":
        #         break

    #    if epoch % 25 == 24:
    #        lr *= 0.5
    #        for g in optimizer.param_groups:
    #            g['lr'] = lr
    #        print("New learning rate: ", lr)


    print('Finished Training')

    PATH = './mnist_dense_net.pt'
    torch.save(net.state_dict(), PATH)
    print("Network saved.")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 1024
    NUM_EPOCHS = 100

    train_data, val_data = create_dataset(train=True, batch_size=BATCH_SIZE)

    net = SmallDenseNet().to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0)

    train(net, train_data, val_data, optimizer, criterion, device, num_epochs=NUM_EPOCHS)
