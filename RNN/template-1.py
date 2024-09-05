# take a look at the csv file yourself first
# columns High, Low, Open are input features and column Close is target value
x = 
y = 

# use StandardScaler from sklearn to standardize
x = scaler_x.fit_transform(...)
y = scaler_y.fit_transform(...)


# split into train and evaluation (8 : 2) using train_test_split from sklearn
train_x, test_x, train_y, test_y = train_test_split(...)


# now make x and y tensors, think about the shape of train_x, it should be (total_examples, sequence_lenth, feature_size)
# we wlll make sequence_length just 1 for simplicity, and you could use unsqueeze at dimension 1 to do this
# also when you create tensor, it needs to be float type since pytorch training does not take default type read using pandas
train_x = torch.tensor(...)
train_y = torch.tensor(...)
test_x = torch.tensor(...) 
seq_len = train_x[0].shape[0] # it is actually just 1 as explained above


# different from CNN which uses ImageFolder method, we don't have such method for RNN, so we need to write the dataset class ourselves, reference tutorial is in the main documentation
class BitCoinDataSet(Dataset):
    def __init__(self, train_x, train_y):
        super(Dataset, self).__init__()
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...


# now prepare the dataloader for training set and evaluation set, and hyperparameters
hidden_size = 
num_layers = 
learning_rate = 
batch_size = 
epoch_size = 

train_dataset = BitCoinDataSet(...)
test_dataset = BitCoinDataSet(...)
train_loader = DataLoader()
test_loader = DataLoader()


# model design goes here
class RNN(nn.Module):

    # there is no "correct" RNN model architecture for this lab either, you can start with a naive model as follows:
    # lstm with 5 layers (or rnn, or gru) -> linear -> relu -> linear
    # lstm: nn.LSTM (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

    def __init__(self, input_feature_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        ...
    
    def forward(self, x):
        ...


# instantiate your rnn model and move to device as in cnn section
rnn = RNN().to(device)
# loss function is nn.MSELoss since it is regression task
criteria = 
# you can start with using Adam as optimizer as well 
optimizer = 



# start training 
rnn.train()
for epoch in range(epoch_size): # start with 10 epochs

    loss = 0.0 # you can print out average loss per batch every certain batches

    for batch_idx, data in enumerate(train_loader):
        # get inputs and target values from dataloaders and move to device
        inputs, targets = 
        inputs.to(device)
        targets.to(device)

        # zero the parameter gradients using zero_grad()

        # forward -> compute loss -> backward propogation -> optimize (see tutorial mentioned in main documentation)

        loss += # add loss for current batch
        if batch_idx % 100 == 99:    # print average loss per batch every 100 batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

print('Finished Training')



prediction = []
ground_truth = []
# evaluation
rnn.eval()
with torch.no_grad():
    for data in test_loader:
        inputs = 
        targets = 
        inputs = inputs.to(device)

        ground_truth += targets.flatten().tolist()
        out = rnn(inputs).detach().cpu().flatten().tolist()
        prediction += out


# remember we standardized the y value before, so we must reverse the normalization before we compute r2score
prediction = scaler_y.inverse_transform(...)
ground_truth = scaler_y.inverse_transform(...)

# use r2_score from sklearn
r2score = r2_score(prediction,ground_truth)