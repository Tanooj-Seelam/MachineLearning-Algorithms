'''
We will use a single-layer perceptron in this lab to learn a simple linear function 3X + 1 = Y where X is a single dimension input variable and Y is a single dimension target variable.  
Here `3` is weight `w` and `1` is bias `b`. 

We will use pytorch in this in class activity, you can use command 'pip install torch' to install torch on your local m ````````````````````achine to develop your code or you can just use colab where is torch is pre-installed.

The code needs you to implement is labeled by TO DO !!!!!, there is not much for you to implement but do learn about the entire workflow.

Name your file slp.py when submitting to GradeScope.
'''

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
        # Here you will need to define a single layer perceptron that takes in 1 dimensional input 
        # variable and output 1 dimensional output variable.
        # nn.Linear(_,_ ): https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc = nn.Linear( , ) # TO DO !!!!!

    # second method you must have is forward() where you actually pass forward 
    # your input features using the layers defined in __init__(). 
    def forward(self, x):
        # This function takes in input variables. 
        # Actaully we can define what to pass in as x with Dataloader and batching, 
        # which will be practiced in lab3. Because sometimes just knowing input features are 
        # not enough for complex models to forward propogate. But this lab, we will simply pass in 
        # input feature itself
        
        # Here, just pass through x with self.fc
        out = # TO DO !!!!!
        
        # Now that we have out, just return this output of a single layer perceptron
        return # TO DO !!!!!


# instantiate a model
net = Net()

# define a optimizer, in this case, we will just use stochastic gradient descent 
# with learning rate of 0.01.
# https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
optimizer = optim.SGD(net.parameters(), lr=) # TO DO !!!!!


# let's create some data, since it is such a simple function, even with 100 data points,
# we will have a decent model, but remember this is not the case in real cases, a model rarely 
# learns something with only 100 input data
X = 2 * torch.rand(100) - 1 # X within range of [-1, 1]
Y = X * 3.0 + 1


# start training, the following workflow is a standard way to training network using pytorch
# we will iterate through 10 epochs
for epoch in range(): # TO DO !!!!!
    epoch_loss = 0 
    for i, (x, y) in enumerate(zip(X, Y)):
        
        # here x is tensor(1.) with dimension 0, y is tensor(4.) with dimension 0
        # however, nn.linear requires data to be at least dimension 1 (for supporting matrix multiplication and batching)
        # we will make tensor(1.) becomes tensor([1.]) first
        # you can use unsqueeze() function 
        # https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        x = torch.unsqueeze(x, ) # TO DO !!!!!
        y = torch.unsqueeze(y, ) # TO DO !!!!!

        # important: set gradient of all model parameters to 0 before calculating gradient 
        # for current example, otherwise pytorch will accumulate current gradient 
        # with previous gradient, this is not a behavior we want in our case
        optimizer.zero_grad()
        
        # forward pass, just pass through x
        output = net( ) # TO DO !!!!!
        
        # calculate loss using model output and target value -> between output and y
        # we will use nn.MSELoss()
        # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        loss = nn.MSELoss()( , ) # TO DO !!!!!

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


