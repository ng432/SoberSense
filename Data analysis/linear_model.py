# %%

import os
from unpacking_data import SoberSenseDataset
from data_loaders import train_loop, test_loop
from data_transforms import sample_transform
import torch as t
from torch import nn
from torch.utils.data import DataLoader, random_split

device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


# creating test
# Starting with just a bog standard linear ReLU network
class linearNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.input_size = input_size

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048), 
            nn.Linear(2048, 1024),  # Adjusted to match the previous batch normalization
            nn.ReLU(),
            nn.BatchNorm1d(1024), 
            nn.Linear(1024, 256),   # Adjusted to match the previous batch normalization
            nn.ReLU(),
            nn.BatchNorm1d(256),  
            nn.Linear(256, 64),     # Adjusted to match the previous batch normalization
            nn.ReLU(),
            nn.BatchNorm1d(64),  
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y


# %%

# test_data has 40 samples, with either 0, 5 or 10 units drunk, and simulated 'drunk behaviour'
data_set = SoberSenseDataset(
    "test_data",
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

input_size = data_set[0][0].flatten().shape[0]

linear_model = linearNetwork(input_size=input_size).to(device)

learning_rate = 3e-3
batch_size = 8

loss_fn = nn.MSELoss()

optimizer = t.optim.SGD(linear_model.parameters(), lr=learning_rate)

epochs = 100
for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train_loop(train_dataloader, linear_model, loss_fn, optimizer)
    test_loop(test_dataloader, linear_model, loss_fn)
print("Done!")


# %%
