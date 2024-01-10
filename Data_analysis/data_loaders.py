

import os
import torch as t
from torch.utils.data import DataLoader, random_split

# Contains functions for loading data


def train_loop(dataloader, model, loss_fn, optimizer, writer, epoch):
    size = len(dataloader.dataset)

    model.train()

    total_train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Creating prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch * len(dataloader) + batch)

        total_train_loss += loss.item()

        if batch % 2 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    total_train_loss = total_train_loss / len(dataloader)

    return total_train_loss

    


def test_loop(dataloader, model, loss_fn):
    model.eval()

    num_batches = len(dataloader)
    test_loss = 0

    with t.no_grad():

        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    return test_loss 