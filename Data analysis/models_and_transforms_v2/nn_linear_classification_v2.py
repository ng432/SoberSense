

#%%

import sys
import os
from data_transforms_v2 import prep_transform, randomly_flipx, randomly_flipy, append_distance, randomly_crop, convert_time_to_intervals, binary_label_transform
import torch as t

from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
sys.path.insert(0, parent_dir)

from data_loaders import train_loop, test_loop
from unpacking_data import SoberSenseDataset
from evaluation_functions import calc_model_prec_recall_f1

device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


#%%

class linear_nn_bc(nn.Module):
    def __init__(self, num_features: int, num_points: int, dropout_prob = 0.0) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.input_size = num_features * num_points

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
    

# %%

def augmentation_transform(x):
    x = randomly_crop(x, crop_size=300)
    x = append_distance(x)
    x = randomly_flipx(x)
    x = randomly_flipy(x)
    x = convert_time_to_intervals(x)
    return x


data_path = os.path.join(parent_dir, 'pilot_data')

data_set = SoberSenseDataset(
    data_path,
    label_name='unitsDrunk', 
    prep_transform=prep_transform,
    augmentation_transform= augmentation_transform,
    label_transform= lambda x: binary_label_transform(x, threshold = 5),
    length_threshold=300
)


#%%
train_ratio = 0.75
test_ratio = 1 - train_ratio

train_size = int(train_ratio * len(data_set))
test_size = len(data_set) - train_size

train_dataset, test_dataset = random_split(data_set, [train_size, test_size])

test_labels = []
test_over_threshold_count = 0 

for i in range(len(test_dataset)):
    label = test_dataset[i][1].item()
    test_labels.append(label)
    test_over_threshold_count += label

train_labels = []
train_over_threshold_count = 0 

for i in range(len(train_dataset)):
    label = train_dataset[i][1].item()
    train_labels.append(label)
    train_over_threshold_count += label

print("Test labels, Over bac threshold count:", test_over_threshold_count)
print("Test labels, under bac threshold count",len(test_labels) - test_over_threshold_count)

print("Train labels, Over bac threshold count:", train_over_threshold_count)
print("Train labels, under bac threshold count",len(train_labels) - train_over_threshold_count)

#%%

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

data_shape = data_set[0][0].shape

linear_model = linear_nn_bc(num_features=data_shape[0], num_points=data_shape[1], dropout_prob=0.1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
learning_rate = 1e-4
optimizer = t.optim.SGD(linear_model.parameters(), lr=learning_rate)

epochs = 2000
writer = SummaryWriter()

#%%
for i in range(epochs):

    print(f"Epoch {i+1}\n-------------------------------")

    train_loss = train_loop(train_dataloader, linear_model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, linear_model, loss_fn)

    writer.add_scalar('Loss/Train', train_loss, i)
    writer.add_scalar('Loss/Validation', test_loss, i)

writer.flush()
print("Done!")


# %%

precision, recall, f1 = calc_model_prec_recall_f1(linear_model, train_dataloader, threshold=0.5)

print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


# %%
