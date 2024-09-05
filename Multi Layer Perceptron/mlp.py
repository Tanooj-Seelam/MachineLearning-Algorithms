import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Define the two linear layers
        self.fc1 = nn.Linear(1, 10)  # 1 input dimension, 10 output dimensions
        self.fc2 = nn.Linear(10, 1)  # 10 input dimensions, 1 output dimension

    def forward(self, x):
        out = self.fc1(x)  # Pass through the first linear layer
        out = F.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Pass through the second linear layer

        return out


net = Net()

# We use SGD optimizer with a learning rate of 0.05
optimizer = optim.SGD(net.parameters(), lr=0.05)  # TO DO !!!!!

# Creating the data: we created 1000 points for quadratic function learning
X = 2 * torch.rand(1000) - 1  # X range is [-1, 1]
Y = X ** 2


for epoch in range(10):
    epoch_loss = 0
    for i, (x, y) in enumerate(zip(X, Y)):
        x = torch.unsqueeze(x, 0)  # Add dimension to match input shape
        y = torch.unsqueeze(y, 0)  # Add dimension to match target shape

        optimizer.zero_grad()

        output = net(x)  # Forward pass

        loss = nn.MSELoss()(output, y)  # Mean squared error loss

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print("Epoch {} - loss: {}".format(epoch + 1, epoch_loss))

# let's test our model
# let's see if we can predict 0.5 * 0.5 = 0.25, 0.3 * 0.3 = 0.09, 0.2 * 0.2 = 0.04
X_test = torch.tensor([0.5, 0.3, 0.2])
Y_test = net(X_test.unsqueeze(1))
Y_test = Y_test.flatten().tolist()
Y_test = [round(y, 2) for y in Y_test]