

#%%


import sys
import os
from data_transforms_v2 import prep_transform, randomly_flipx, randomly_flipy, append_distance, randomly_crop, convert_time_to_intervals
import torch as t

from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
sys.path.insert(0, parent_dir)

from data_loaders import train_loop, test_loop
from unpacking_data import SoberSenseDataset

device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


#%%

class linear_nn_bc(nn.Module):
    def __init__(self, num_features: int, num_points: int) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.input_size = num_features * num_points
        self.sigmoid = nn.Sigmoid()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), 
            nn.ReLU(),
            nn.Linear(1024, 256),   
            nn.BatchNorm1d(256),    
            nn.ReLU(),
            nn.Linear(256, 64),   
            nn.BatchNorm1d(64),    
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        y = self.sigmoid(x)
        return y
    

# %%

def augmentation_transform(x):
    x = randomly_crop(x, crop_size=250)
    x = append_distance(x)
    x = randomly_flipx(x)
    x = randomly_flipy(x)
    x = convert_time_to_intervals(x)
    return x

def binary_label_transform(label, threshold = 0.08):

    # 1 represents 
    if label > threshold:
        label = t.tensor([1], dtype=t.float32).to(device)
    else:
        label = t.tensor([0], dtype=t.float32).to(device)

    return label

sample_data_path = os.path.join(parent_dir, 'pilot_data')

data_set = SoberSenseDataset(
    sample_data_path,
    prep_transform=prep_transform,
    augmentation_transform= augmentation_transform,
    label_transform= binary_label_transform
)

#%%
train_ratio = 0.6
test_ratio = 1 - train_ratio

train_size = int(train_ratio * len(data_set))
test_size = len(data_set) - train_size

train_dataset, test_dataset = random_split(data_set, [train_size, test_size])

test_labels = []
over_bac_threshold_count = 0 

for i in range(len(test_dataset)):
    label = test_dataset[i][1].item()
    test_labels.append(label)
    over_bac_threshold_count += label

print("Test labels, Over bac threshold count:", over_bac_threshold_count)
print("Test labels, under bac threshold count",len(test_labels) - over_bac_threshold_count)

#%%

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

data_shape = data_set[0][0].shape

linear_model = linear_nn_bc(num_features=data_shape[0], num_points=data_shape[1]).to(device)
loss_fn = nn.BCELoss()
learning_rate = 3e-4
optimizer = t.optim.SGD(linear_model.parameters(), lr=learning_rate)

epochs = 2000
writer = SummaryWriter()
for i in range(epochs):

    print(f"Epoch {i+1}\n-------------------------------")

    train_loss = train_loop(train_dataloader, linear_model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, linear_model, loss_fn)

    writer.add_scalar('Loss/Train', train_loss, i)
    writer.add_scalar('Loss/Validation', test_loss, i)


writer.flush()
print("Done!")


# %%

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, threshold = 0.6):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predictions = []

    with t.no_grad():  
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted_probs = outputs.squeeze().cpu() 
            predicted_labels = (predicted_probs >= threshold).float() 
            true_labels.append(y.cpu())
            predictions.append(predicted_labels)

    true_labels = np.concatenate(true_labels)
    predictions = np.concatenate(predictions)

    print([predictions[i] for i in range(len(true_labels))])
    print([true_labels[i][0] for i in range(len(true_labels))])

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return precision, recall, f1

# Evaluate the model
precision, recall, f1 = evaluate_model(linear_model, test_dataloader)
print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


# %%
