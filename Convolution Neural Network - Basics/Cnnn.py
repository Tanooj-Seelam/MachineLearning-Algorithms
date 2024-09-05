import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Input channels: 1, Output channels: 8, Kernel size: 3
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        # Kernel size: 2
        self.pool = nn.AvgPool2d(kernel_size=2)
        # Given initial input size (100, 100), after convolution with kernel size 3 and pooling with kernel size 2,
        # the resulting size will be (49, 49) due to floor division
        self.fc_input_features = 8 * 49 * 49

        # Define the fully connected (linear) layer
        self.fc = nn.Linear(self.fc_input_features, 10)  # 10 output classes

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.pool(out)
        out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layer
        out = self.fc(out)
        return out


# Testing the model and checking the shapes
model = CNN()
x = torch.randn(20, 1, 100, 100)  # 20 images, single channel, width and height both 100
out_shape = list(model(x).shape)  # Output shape should be (20, 10)

# Checking the shape of the feed forward layer (self.fc) in the network
for name, param in model.named_parameters():
    if name == "fc.weight":
        fc_shape = list(param.shape)

