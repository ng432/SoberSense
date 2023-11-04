

import os
import torch as t
from torch.utils.data import DataLoader, random_split


# Returns tensor of shape:  [Ch, D, S]
# Ch (=2): channels, one representing touch data, one representing animation path of circle
# D (=3): x coordinate, y coordinate, time
# S: Number of samples of animation path (including those interpolated). touchData padded with 0s to match animation path length
def sample_transform(sample_data, device='mps'):
    path_data = t.tensor(
        [sample_data["randomPath"]["X"], sample_data["randomPath"]["Y"], sample_data["randomPath"]["time"]]
    )

    touch_data = t.tensor(
        [sample_data["touchData"]["X"], sample_data["touchData"]["Y"], sample_data["touchData"]["time"]]
    )

    path_length = path_data.shape[1]
    td_length = touch_data.shape[1]

    if td_length > path_length:
        raise AssertionError(
            f"Touch data (length={td_length})  is longer than the animation path data (length={path_length}). Try increase the number of interpolation points in the animation path."
        )

    padding_length = path_length - td_length

    front_pad = padding_length // 2
    back_pad = padding_length - front_pad

    padded_td = t.nn.functional.pad(touch_data, (front_pad, back_pad))

    sample_tensor = t.stack((padded_td, path_data))

    # MPS requires float32
    if device == "mps":
        sample_tensor = sample_tensor.to(t.float32)
        sample_tensor = sample_tensor.to(device)

    return sample_tensor



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Creating prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 4 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()

    num_batches = len(dataloader)
    test_loss = 0

    with t.no_grad():

        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    # correct /= size
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")