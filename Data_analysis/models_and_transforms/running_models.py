# %%

import sys
import os
from data_transforms import (
    processing_transform,
    randomly_flipx,
    randomly_flipy,
    append_distance,
    randomly_crop,
    convert_time_to_intervals,
    binary_label_transform,
    append_RT,
    append_velocity_and_acceleration,
)
import torch as t
import json

from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)
sys.path.insert(0, parent_dir)

from data_loaders import train_loop, test_loop
from unpacking_data import SoberSenseDataset
from evaluation_functions import calc_prec_recall_f1, calc_prf1_majority_vote
from nn_models import LSTM_binary_classifier, ConvNN, linear_nn_bc

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

with open("av_normalising_values.json", "r") as file:
    av_normalising_values = json.load(file)


# %%

crop_size = 400


# prep transform is cached
def prep_transform(unprocessed_data):
    x = processing_transform(unprocessed_data)
    x = append_distance(x)
    x, _ = append_RT(x, unprocessed_data, normalising=True)
    x = append_velocity_and_acceleration(x, norm_dic=av_normalising_values)
    return x


def augmentation_transform(x):
    x = randomly_crop(x, crop_size=crop_size)
    # x = convert_time_to_intervals(x) # ideally would be in prep transform, but has to be after cropping
    x = randomly_flipx(x)
    x = randomly_flipy(x)
    return x


data_path = os.path.join(parent_dir, "pilot_data")

data_set = SoberSenseDataset(
    data_path,
    label_name="BAC",
    prep_transform=prep_transform,
    augmentation_transform=augmentation_transform,
    label_transform=lambda x: binary_label_transform(x, threshold=0.08),
    length_threshold=crop_size,
)


# %%
train_ratio = 0.8
test_ratio = 1 - train_ratio

train_size = int(train_ratio * len(data_set))
test_size = len(data_set) - train_size

train_dataset, test_dataset = random_split(data_set, [train_size, test_size])

test_labels = []
test_over_threshold_count = 0

max_values = []
min_values = []

# Note first pass of data set requires a relatively high amount of computation
for i in range(len(test_dataset)):
    print(f"\rTest set: Processing {i+1} data point of {len(test_dataset)}", end="")
    label = test_dataset[i][1].item()
    test_labels.append(label)
    test_over_threshold_count += label
    max_values.append(test_dataset[i][0].max().item())
    min_values.append(test_dataset[i][0].min().item())

train_labels = []
train_over_threshold_count = 0

for i in range(len(train_dataset)):
    print(f"\rTrain set: Processing {i+1} data point of {len(train_dataset)}", end="")
    label = train_dataset[i][1].item()
    train_labels.append(label)
    train_over_threshold_count += label
    max_values.append(train_dataset[i][0].max().item())
    min_values.append(train_dataset[i][0].min().item())

print("\nMax value from data", max(max_values))
print("Min value from data", max(min_values))
print("-------------------------------")
print("Test labels, Over bac threshold count:", test_over_threshold_count)
print(
    "Test labels, under bac threshold count",
    len(test_labels) - test_over_threshold_count,
)
print("-------------------------------")
print("Train labels, Over bac threshold count:", train_over_threshold_count)
print(
    "Train labels, under bac threshold count",
    len(train_labels) - train_over_threshold_count,
)
print("-------------------------------")

positive_weight = (
    len(train_labels) - train_over_threshold_count
) / train_over_threshold_count

print(f"Positive weight for loss: {positive_weight:.2f}")


# %%

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

data_shape = data_set[0][0].shape

model = ConvNN(num_features=data_shape[0]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=t.tensor(positive_weight, device=device))
learning_rate = 1e-5
# optimizer = t.optim.SGD(linear_model.parameters(), lr=learning_rate)
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 5000
writer = SummaryWriter()

# %%
for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")

    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, writer, i)
    test_loss = test_loop(test_dataloader, model, loss_fn)

    writer.add_scalar("Loss/Train", train_loss, i)
    writer.add_scalar("Loss/Validation", test_loss, i)


writer.flush()
print("Done!")

# %%

# saving model and data sets to recreate

t.save(model.state_dict(), "cnn_with_bac.pth")


# %%

precision, recall, f1, true_labels, predicted_labels = calc_prf1_majority_vote(
    model, test_dataloader, threshold=0.5, num_rep=1002
)

print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

# %%

for i in range(len(true_labels)):
    print(f"Predicted label: {predicted_labels[i]}, True label: {true_labels[i]}")
