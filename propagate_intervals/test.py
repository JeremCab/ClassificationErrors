import torch
import tqdm

from propagate_intervals.dataset_old import create_dataset
from network import SimpleNet, DenseNet, SmallDenseNet, SmallConvNet

BATCH_SIZE = 256

# load data
test_data = create_dataset(train=False, batch_size=BATCH_SIZE)

# create network 
net = SmallDenseNet()
PATH="./mnist_dense_net.pt"
net.load_state_dict(torch.load(PATH))
net.eval()

net = net.cuda()#.half()
print(net)

correct = 0
num = 0
for data in tqdm.tqdm(test_data):
    inputs, labels = data
    inputs = inputs.cuda()#.half()
    labels = labels.cuda()

    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    num += labels.size(0)
    correct += (predicted == labels).sum().item()

print(correct)
print(f"acc: {100*correct/num:.3f}")



# =================== #
# *** NEW VERSION *** #
# =================== #


def evaluate(net, test_loader, criterion, device):
    net.to(device)
    net.eval()   # set to evaluation mode
    
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
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


# load best saved model if needed
net.load_state_dict(torch.load("best_model.pth"))

# evaluate
test_loss, test_acc = evaluate(net, test_loader, criterion, device)
