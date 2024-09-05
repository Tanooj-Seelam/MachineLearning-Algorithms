import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

# Load the data
data = pd.read_csv('./coin_Bitcoin.csv')
x = data[['High', 'Low', 'Open']].values
y = data['Close'].values

# Standardize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Split into training and evaluation sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=40)

# Convert data to PyTorch tensors
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)

# Define the RNN dataset class
class BitCoinDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Create data loaders
batch_size = 64
train_dataset = BitCoinDataSet(train_x, train_y)
test_dataset = BitCoinDataSet(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_feature_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the RNN model
input_feature_size = 3
hidden_size = 128
num_layers = 5
rnn = RNN(input_feature_size, hidden_size, num_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rnn.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.001)

rnn.train()
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Ensure the input has the correct shape for LSTM (batch_size, sequence_length, input_size)
        inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()
        outputs = rnn(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print('Finished Training')

# Evaluation
rnn.eval()
prediction = []
ground_truth = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)

        # Ensure the input has the correct shape for LSTM (batch_size, sequence_length, input_size)
        inputs = inputs.unsqueeze(1)

        out = rnn(inputs).cpu().numpy()
        prediction.extend(out)
        ground_truth.extend(targets)

# Reverse the normalization
prediction = scaler_y.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
ground_truth = scaler_y.inverse_transform(np.array(ground_truth).reshape(-1, 1)).flatten()

# Calculate the R2 score
r2score = r2_score(ground_truth, prediction)