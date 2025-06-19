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
