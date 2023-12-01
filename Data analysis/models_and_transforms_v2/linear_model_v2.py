
# %% 

import sys
import os
from data_transforms_v2 import prep_transform, randomly_flipx, randomly_flipy, append_distance, randomly_crop
import torch as t

from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
sys.path.insert(0, parent_dir)

from data_loaders import train_loop, test_loop
from unpacking_data import SoberSenseDataset

device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# %%


class linearNetwork(nn.Module):
    def __init__(self, num_features: int, num_points: int) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.input_size = num_features * num_points

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048), 
            nn.Linear(2048, 1024),     
            nn.ReLU(),
            nn.BatchNorm1d(1024),  
            nn.Linear(1024, 256),     
            nn.ReLU(),
            nn.BatchNorm1d(256),  
            nn.Linear(256, 64),     
            nn.ReLU(),
            nn.BatchNorm1d(64),  
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y
# %%

sample_data_path = os.path.join(parent_dir, 'sample_data')

def augmentation_transform(x):
    x = randomly_crop(x, crop_size=100)
    x = append_distance(x)
    x = randomly_flipx(x)
    x = randomly_flipy(x)
    return x


# test_data has 40 samples, with either 0, 5 or 10 units drunk, and simulated 'drunk behaviour'
data_set = SoberSenseDataset(
    sample_data_path,
    prep_transform=prep_transform,
    augmentation_transform= augmentation_transform,
    label_transform=lambda x: t.tensor(x, dtype=t.float32).to(device)
)

# %%

train_ratio = 0.7
test_ratio = 1 - train_ratio

train_size = int(train_ratio * len(data_set))
test_size = len(data_set) - train_size

train_dataset, test_dataset = random_split(data_set, [train_size, test_size])

batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

data_shape = data_set[0][0].shape

linear_model = linearNetwork(num_features=data_shape[0], num_points=data_shape[1]).to(device)

learning_rate = 2e-3

loss_fn = nn.MSELoss()

optimizer = t.optim.SGD(linear_model.parameters(), lr=learning_rate)

epochs = 1000
writer = SummaryWriter()
for i in range(epochs):

    print(f"Epoch {i+1}\n-------------------------------")

    train_loss = train_loop(train_dataloader, linear_model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, linear_model, loss_fn)

    writer.add_scalar('Loss/Train', train_loss, i)
    writer.add_scalar('Loss/Validation', test_loss, i)


writer.flush()
print("Done!")


