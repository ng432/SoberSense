
#%% 

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

class TransformerRegressor(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden, n_layers, dropout):
        super().__init__()

        self.encoder = nn.Linear(n_features, n_hidden)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden, nhead=n_heads, dropout=dropout
        )
        
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        self.decoder = nn.Linear(n_hidden, 1)

    def forward(self, x):

        # input is of shape [B, num_features, seq_length]
        # need to rearrange to [B, seq_length, num_features]

        # this will be encoded to [B, seq_length, n_hidden ]
        x = t.transpose(x, -1, -2)
        x = self.encoder(x)

        x = self.transformer(x)
        x = self.decoder(x[-1])
        return x
    
#%%

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

#%%

data_shape = data_set[0][0].shape
n_features = data_shape[0]

n_heads = 12
n_hidden = 128
n_layers = 6
dropout = 0.1

model = TransformerRegressor(n_features, n_heads, n_hidden, n_layers, dropout).to(device)

# %%


train_ratio = 0.6
test_ratio = 1 - train_ratio
train_size = int(train_ratio * len(data_set))
test_size = len(data_set) - train_size
train_dataset, test_dataset = random_split(data_set, [train_size, test_size])

batch_size = 4

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

loss_fn = nn.MSELoss()

learning_rate = 3e-3
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 1000
writer = SummaryWriter()

for i in range(epochs):

    print(f"Epoch {i+1}\n-------------------------------")

    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, model, loss_fn)

    writer.add_scalar('Loss/Train', train_loss, i)
    writer.add_scalar('Loss/Validation', test_loss, i)


writer.flush()
print("Done!")



# %%
