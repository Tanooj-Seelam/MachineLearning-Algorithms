# import torch and other necessary modules from torch
import torch
import torch.nn as nn
...
# import torchvision and other necessary modules from torchvision 
import torchvision
...


# recommended preprocessing steps: resize to square -> convert to tensor -> normalize the image
# if you are resizing, 100 is a good choice otherwise GradeScope will time out
# you could use Compose (https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html) from transforms module to handle preprocessing more conveniently
transform = transforms.Compose(...)


# thanks to torchvision, this is a convenient way to read images from folders directly without writing datasets class yourself (you should know what datasets class is as mentioned in the documentation)
dataset = datasets.ImageFolder(...)


# now we need to split the data into training set and evaluation set 
# use 20% of the dataset as test
test_set, train_set = torch.utils.data.random_split(...)
# prepare dataloader for training set and evaluation set
trainloader = torch.utils.data.DataLoader(...)
testloader = ...

# model hyperparameter
learning_rate = 
batch_size = 
epoch_size = 



# model design goes here
class CNN(nn.Module):

    # there is no "correct" CNN model architecture for this lab, you can start with a naive model as follows:
    # convolution -> relu -> pool -> convolution -> relu -> pool -> convolution -> relu -> pool -> linear -> relu -> linear -> relu -> linear
    # you can try increasing number of convolution layers or try totally different model design
    # convolution: nn.Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    # pool: nn.MaxPool2d (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    # linear: nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

    def __init__(self):
        super(CNN,self).__init__()
        ...

    def forward(self, x):



device = 'cuda' if torch.cuda.is_available() else 'cpu' # whether your device has GPU
cnn = CNN().to(device) # move the model to GPU
# search in official website for CrossEntropyLoss
criterion = 
# try Adam optimizer (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with learning rate 0.0001, feel free to use other optimizer
optimizer = 


# start model training
cnn.train() # turn on train mode, this is a good practice to do
for epoch in range(epoch_size): # begin with trying 10 epochs 

    loss = 0.0 # you can print out average loss per batch every certain batches

    for i, data in enumerate(trainloader, 0):
        # get the inputs and label from dataloader
        inputs, label = 
        # move tensors to your current device (cpu or gpu)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients using zero_grad()

        # forward -> compute loss -> backward propogation -> optimize (see tutorial mentioned in main documentation)

        # print some statistics
        loss +=  # add loss for current batch 
        if i % 100 == 99:    # print out average loss every 100 batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

print('Finished Training')


# evaluation on evaluation set
ground_truth = []
prediction = []
cnn.eval() # turn on evaluation model, also a good practice to do
with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs, so turn on no_grad mode
    for data in testloader:
        inputs, labels = 
        inputs = inputs.to(device)
        ground_truth += # convert labels to list and append to ground_truth
        # calculate outputs by running inputs through the network
        outputs = cnn(inputs)
        # the class with the highest logit is what we choose as prediction
        _, predicted = torch.max(...)
        prediction += ... # convert predicted to list and append to prediction


# GradeScope is chekcing for these three variables, you can use sklearn to calculate the scores
accuracy = accuracy_score(...)
recall = recall_score(...)
precision = precision_score(...)






