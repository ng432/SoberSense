
#%% 


import os
from unpacking_data import SoberSenseDataset
from data_loaders import sample_transform, train_loop, test_loop
import torch as t
from torch import nn
from torch.utils.data import DataLoader, random_split

device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print(f"Using {device} device")



#%%

class regressionCNN(nn.Module):
    def __init__(self, input_channels, num_samples):
        super(regressionCNN, self).__init__()

        # Size of sample data is B, 2, 3, 1525
        # B: batch size 
        # 2 = input_channel, animation path or touch data
        # 3: x, y or timestamp
        # 1525 = sample_number number of data points

        self.ReLU = nn.ReLU()

        conv1_ks = (1,10)

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size = conv1_ks, padding = 'same')
        # Sample data: will give shape (B, 32, 3, 1525)

        # will extract along the time dimension 
        maxpool1_ks = (1,25)

        # Sample data: input (B, 32, 3, 1525)
        self.maxpool1 = nn.MaxPool2d(kernel_size = maxpool1_ks)
        # Sample data: output (B, 32, 3, 1525/maxpool1_ks[1])

        # will combine across x, y and time stamp
        conv2_ks = (3, 3)

        # Sample data: input (B, 32, 3, 1525/maxpool1_ks[1])
        self.conv2 = nn.Conv2d(32, 64, kernel_size = conv2_ks)
        # Sample data: output (B, 64, 1, 1525/maxpool1_ks[1] - (conv2_ks[1] - 1))

        self.flatten = nn.Flatten()

        # size of resulting last dim from conv2
        conv2_result_dim = (num_samples / maxpool1_ks[1]) - (conv2_ks[1] - 1)
        # Sample data: will equal 59

        self.fc1 = nn.Linear(64 * int(conv2_result_dim), 1024)

        self.fc2 = nn.Linear(1024, 256)
        
        self.fc3 = nn.Linear(256, 1)

        

    def forward(self, x):

        # First convolutional layer
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.maxpool1(x)

        # Second convolutional layer
        x = self.conv2(x)
        x = self.ReLU(x)

        # gives shape of (B, 64, 59)
        x = t.squeeze(x)
        
        # gives shape of (B, 64 * 59)
        x = self.flatten(x)

        # outputs shape of (B, 128)
        x = self.fc1(x)
        x = self.ReLU(x)

        x = self.fc2(x)
        x = self.ReLU(x)
        
        x = self.fc3(x)

        return x
    
#%%

data_set = SoberSenseDataset(
    "sample_data",
    sample_transform=sample_transform,
    label_transform=lambda x: t.tensor(x, dtype=t.float32).to(device),
    animation_interp_number=60,
)

train_ratio = 0.8
test_ratio = 1 - train_ratio

train_size = int(train_ratio * len(data_set))
test_size = len(data_set) - train_size

train_dataset, test_dataset = random_split(data_set, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4)
test_dataloader = DataLoader(test_dataset, batch_size=4)

#%%

# test_data has 40 samples, with either 0, 5 or 10 units drunk, and simulated 'drunk behaviour'


learning_rate = 6e-3
batch_size = 8

loss_fn = nn.MSELoss()

# number of samples per recordin 
num_samples = data_set[0][0].shape[2]

cnn_model = regressionCNN(
    input_channels = 2, 
    num_samples = num_samples).to(device)

optimizer = t.optim.SGD(cnn_model.parameters(), lr=learning_rate)

#%%

epochs = 200
for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train_loop(train_dataloader, cnn_model, loss_fn, optimizer)
    test_loop(test_dataloader, cnn_model, loss_fn)
print("Done!")




# %%
