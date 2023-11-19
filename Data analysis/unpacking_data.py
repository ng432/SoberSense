# %%

# Code to unpack data recorded from the SoberSense app

import json
import os
import torch as t
from torch.utils.data import Dataset
import numpy as np


class SoberSenseDataset(Dataset):

    def __init__(
        self, data_folder, device="mps", prep_transform=None, augmentation_transform=None, label_transform=None):

        if augmentation_transform is not None and prep_transform is None:
            raise ValueError("Augmentation transform provided without a preparatory transform.")

        self.data_folder = data_folder
        self.device = device
        self.prep_transform = prep_transform
        self.prep_cache = {} # prep transform can be expensive to compute, so cache to store results
        self.augmentation_transform = augmentation_transform
        self.label_transform = label_transform
        self.samples = []  # A list to store the data examples
        
        # Iterate through JSON files in the data folder
        for filename in os.listdir(data_folder):
            if filename.endswith(".json"):
                with open(os.path.join(data_folder, filename), "r") as json_file:

                    sample_data = json.load(json_file)

                    if "duration" in sample_data:
                        # time, in seconds, that the game runs over
                        game_length = sample_data["duration"]
                    # for sample_data, where data doesn't have 'duration', but was 20 seconds
                    else:
                        game_length = 20

                    scale = {
                        "x_scale": sample_data["screenSize"]["width"],
                        "y_scale": sample_data["screenSize"]["height"],
                    }

                    # This will output scaled touch data (from 0 to 1)
                    unpacked_touchData = UnpackCoordData(
                        sample_data["touchData"],
                        "xLocation",
                        "yLocation",
                        start_time = sample_data["randomPath"][0]["time"],
                        scale = scale,
                        game_length = game_length,
                    )

                    # Random path data is recorded as scaled, but from -0.5 to 0.5
                    # Hence, shift = 0.5 to give normalised coordinates from 0 to 1
                    unpacked_randomPath = UnpackCoordData(
                        sample_data["randomPath"],
                        "xDimScale",
                        "yDimScale",
                        start_time=sample_data["randomPath"][0]["time"],
                        shift=0.5,
                        game_length=game_length,
                    )

                    label = sample_data["unitsDrunk"]

                    sample = {"touchData": unpacked_touchData,
                               "randomPath": unpacked_randomPath,
                               "controlPoints": sample_data["controlPoints"],
                               "animationDuration": sample_data["animationDuration"],
                               "gameLength": game_length}
                    
                    self.samples.append((sample, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        if idx in self.prep_cache: # if already applied prep transform
            sample_data = self.prep_cache[idx]
        elif self.prep_transform: # if haven't applied prep transform, and it exists
            sample_data, _ = self.samples[idx]
            sample_data = self.prep_transform(sample_data)
            self.prep_cache[idx] = sample_data
        else: # if no prep transform
            sample_data, _ = self.samples[idx]

        _ , label = self.samples[idx]

        if self.augmentation_transform:
            sample_data = self.augmentation_transform(sample_data)
            sample_data = sample_data.to(self.device)

        if self.label_transform:
            label = self.label_transform(label)
            label = label.to(self.device)

        return sample_data, label


def UnpackCoordData(data, x_name, y_name, start_time, scale=None, shift=0, game_length=1):
    # If scale provided, scale the touch data by width or height

    x = [point[x_name] / scale["x_scale"] if scale else point[x_name] + shift for point in data]
    y = [point[y_name] / scale["y_scale"] if scale else point[y_name] + shift for point in data]

    time = [(point["time"] - start_time) / game_length for point in data]

    unpacked_data = {"X": x, "Y": y, "time": time}

    return unpacked_data





# %%
