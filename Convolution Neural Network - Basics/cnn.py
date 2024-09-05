'''
In this lab, we will write a CNN model.  
Name this file cnn.py.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(42)


# Suppose our model takes in input with shape (batch_size, channel_size, height, width), we need to classify each input example into 10 classes
# we know input channel_size is 1, height and width are both 100
# we want our architecture to be:
# x -> Conv2d -> relu -> AvgPool2d -> Linear
# nn.AvgPool2d: https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
# nn.Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        # we know input channel size is 1, we want output channel size to be 8 and kernel size to be 3
        self.conv = nn.Conv2d( ) # TO DO !!!!!
        # we want kernel size to be 2
        self.pool = nn.AvgPool2d( ) # TO DO !!!!!
        # think about what is the input feature size of Linear function here with the forward function below
        self.fc = nn.Linear( , ) # TO DO !!!!!
    
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.pool(out)
        out = out.reshape(out.shape[0], -1) # linearalize each input, for example, we want shape (10, 3, 20, 20) to be (10, 3 * 20 * 20) if batch size is 10,
        out = self.fc(out)

        return out


# we are checking the correctness of your CNN with some data
model = CNN()
x = torch.randn(20, 1, 100, 100) # 20 images, single channel, width and height both 100
out_shape = list(model(x).shape) # your output shape should be (20, 10)

# we are also checking the shape of the feed forward layer (self.fc) in the network 
for name, param in model.named_parameters():
    if name == "fc.weight":
        fc_shape = list(param.shape)