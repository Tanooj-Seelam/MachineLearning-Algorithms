import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Define the transformation steps in the Compose pipeline
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize to a square (100x100)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Define the dataset using the ImageFolder and apply the transform
dataset = datasets.ImageFolder(root='./petimages', transform=transform)

# Define the proportion of data for the test set
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size

# Split the dataset into train and test sets
train_set, test_set = random_split(dataset, [train_size, test_size])

# Define batch size and create DataLoader for the training and test sets
batch_size = 32  # Adjust the batch size as needed
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

learning_rate = 0.001  # Adjust as needed
batch_size = 32  # Already defined earlier
epoch_size = 10  # Adjust as needed (number of training epochs)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # num_classes is the number of output classes

    def forward(self, x):
        # Apply the convolutional layers, ReLU, and max-pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor
        x = x.view(-1, 64 * 12 * 12)

        # Apply the fully connected layers and ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move the model to the appropriate device
cnn = CNN().to(device)

# CrossEntropyLoss is commonly used for multi-class classification problems
criterion = nn.CrossEntropyLoss()

# Adam optimizer with a learning rate of 0.0001
learning_rate = 0.0001
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

cnn.train()  # Turn on train mode

for epoch in range(epoch_size):  # Iterate through epochs
    running_loss = 0.0  # Initialize running loss for this epoch

    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels from the dataloader
        inputs, labels = data

        # Move tensors to the current device (CPU or GPU)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = cnn(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print average loss every 100 batches
            print(f'Epoch [{epoch + 1}, Batch {i + 1}] - Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Lists to store ground truth and predictions
ground_truth = []
prediction = []

cnn.eval()  # Turn on evaluation mode
with torch.no_grad():  # Disable gradient calculation

    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)

        # Append ground truth labels to the list
        ground_truth += labels.cpu().numpy().tolist()

        # Calculate outputs by running inputs through the network
        outputs = cnn(inputs)

        # Get the class predictions
        _, predicted = torch.max(outputs, 1)

        # Append predictions to the list
        prediction += predicted.cpu().numpy().tolist()

# Calculate evaluation metrics
accuracy = accuracy_score(ground_truth, prediction)
recall = recall_score(ground_truth, prediction, average='weighted')
precision = precision_score(ground_truth, prediction, average='weighted')
