import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)


class RNN(nn.Module):
    def __init__(self, input_feature_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=input_feature_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True)  # TO DO !!!!!

        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)  # TO DO !!!!!

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initialize h0 with zeros

        out, h_last = self.rnn(x, h0)  # TO DO !!!!!

        out = out.reshape(x.size(0), -1)  # Flatten the output
        out = self.fc(out)  # Pass it through the linear layer

        return out  # (batch_size, num_classes)


x = torch.randn(10, 5, 512)  # 10 sequences, 5 time steps for each sequence, each time step has 512 features

input_feature_size = 512  # TO DO !!!!!
hidden_size = 1024  # TO DO !!!!!
num_layers = 2  # TO DO !!!!!
num_classes = 3  # TO DO !!!!!
sequence_length = 5  # TO DO !!!!!

model = RNN(input_feature_size, hidden_size, num_layers, num_classes, sequence_length)
out_shape = list(model(x).shape)
print(out_shape)

for name, param in model.named_parameters():
    if name == "fc.weight":
        fc_shape = list(param.shape)
print(fc_shape)
