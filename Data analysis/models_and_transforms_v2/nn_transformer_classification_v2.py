
#%% 

import sys
import os
from data_transforms_v2 import prep_transform, randomly_flipx, randomly_flipy, append_distance, randomly_crop, binary_label_transform
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

class TransformerBC(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden, n_layers, dropout):
        super().__init__()

        self.encoder = nn.Linear(n_features, n_hidden)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden, nhead=n_heads, dropout=dropout
        )
        
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        self.decoder = nn.Linear(n_hidden, 1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # input is of shape [B, num_features, seq_length]
        # need to rearrange to [B, seq_length, num_features]

        # this will be encoded to [B, seq_length, n_hidden ]
        x = t.transpose(x, -1, -2)
        x = self.encoder(x)

        x = t.transpose(x, 0, 1)
        
        x = self.transformer(x)

        x = self.decoder(x[-1])

        x = self.sigmoid(x)
  
        
        return x
    
#%%


def augmentation_transform(x):
    x = randomly_crop(x, crop_size=250)
    x = append_distance(x)
    x = randomly_flipx(x)
    x = randomly_flipy(x)
    return x

def binary_label_transform(label, threshold = 0.07):
    # 1 represents 
    if label > threshold:
        label = t.tensor([1], dtype=t.float32).to(device)
    else:
        label = t.tensor([0], dtype=t.float32).to(device)

    return label

data_path = os.path.join(parent_dir, 'pilot_data')

# test_data has 40 samples, with either 0, 5 or 10 units drunk, and simulated 'drunk behaviour'
data_set = SoberSenseDataset(
    data_path,
    label_name='BAC', 
    prep_transform=prep_transform,
    augmentation_transform= augmentation_transform,
    label_transform=binary_label_transform,
    length_threshold=300
)

#%%

data_shape = data_set[0][0].shape
n_features = data_shape[0]

n_heads = 12
n_hidden = 144
n_layers = 6
dropout = 0.1

model = TransformerBC(n_features, n_heads, n_hidden, n_layers, dropout).to(device)


# %%


train_ratio = 0.75
test_ratio = 1 - train_ratio
train_size = int(train_ratio * len(data_set))
test_size = len(data_set) - train_size
train_dataset, test_dataset = random_split(data_set, [train_size, test_size])


#%%

batch_size = 8

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

loss_fn = nn.BCELoss()

learning_rate = 3e-3
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 60000
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
