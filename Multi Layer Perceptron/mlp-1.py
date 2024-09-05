'''
We will use a simple multi-layer perceptron network in this lab to learn a quadratic function X^2 = Y` where `x` is a single dimension input variable and Y is a single dimension target variable.  
Name this file mlp.py.

We will use pytorch in this in-class activity, you can use command 'pip install torch' to install torch on your local machine to develop your code or you can just use colab where torch is pre-installed.

The code needs to be implemented is labeled TO DO !!!!!, there is not much for you to implement but do learn about the entire workflow.

Name your file mlp.py when submitting to GradeScope.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(42)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Here you will need to define two single-layer perceptrons (with non-linearality) to form a two-layer perceptron network.
        # Input is 1 dimension, we will make the intermediate dimension to be 10, the final output dimension is also 1. Below is the network:
        # X (1 dimension) -> 10 dimensional internal layer -> Relu -> Y (1 dimension)
        # We use nn.Linear for a single-layer perceptron as in the previous ICE.
        # For ReLU implementation, we will use F.relu (https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html)
        self.fc1 =  # TO DO !!!!!
        self.fc2 =  # TO DO !!!!!

    def forward(self, x):
        out =  # TO DO !!!!! pass through self.fc1
        out =  # TO DO !!!!! pass through F.relu
        out =  # TO DO !!!!! pass through self.fc2

        return # TO DO !!!!!


net = Net()

# We use SGD optimizer with a learning rate of 0.05
optimizer = optim.SGD(net.parameters(), ) # TO DO !!!!!

# Creating the data: we created 1000 points for quadratic function learning
X = 2 * torch.rand(1000) - 1 # X range is [-1, 1]
Y = X ** 2

# train 10 epochs
for epoch in range( ): # TO DO !!!!!
    epoch_loss = 0 
    for i, (x, y) in enumerate(zip(X, Y)):

        x = torch.unsqueeze(x, ) # TO DO !!!!!
        y = torch.unsqueeze(y, ) # TO DO !!!!!

        optimizer.zero_grad()

        output = net( ) # TO DO !!!!!

        loss = nn.MSELoss()( , ) # TO DO !!!!!

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
