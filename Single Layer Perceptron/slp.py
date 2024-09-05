import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# for reproducibility, let's all use same random seed
torch.manual_seed(42)


# define a class that inherits nn.Module, and inside this class, you will write your model.
class Net(nn.Module):

    # first method you must have is __init__() where various model layers are defined.
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        out = self.fc(x)  # TO DO: Pass input through the linear layer
        return out


# instantiate a model
net = Net()

# Define optimizer and learning rate
optimizer = optim.SGD(net.parameters(), lr=0.01)

X = 2 * torch.rand(100) - 1 # X within range of [-1, 1]
Y = X * 3.0 + 1

for epoch in range(10):
    epoch_loss = 0
    for i, (x, y) in enumerate(zip(X, Y)):
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        optimizer.zero_grad()
        output = net(x)
        loss = nn.MSELoss()(output, y)
        # backward propogate
        loss.backward()

        # update parameters with one optimization step
        optimizer.step()

        # add up loss for current example to the total loss for current epoch
        epoch_loss += loss.item()

    # print out some statistics
    print("Epoch {} - loss: {}".format(epoch + 1, epoch_loss))

# now let's check what is learned
for name, param in net.named_parameters():
    if "weight" in name:
        w = round(param.item(), 1)
    if "bias" in name:
        b = round(param.item(), 1)