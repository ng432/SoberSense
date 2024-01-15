# %%

# Code to unpack data recorded from the SoberSense app

import json
import os
from torch.utils.data import Dataset


class SoberSenseDataset(Dataset):
    def __init__(
        self,
        data_folder,
        label_name="unitsDrunk",
        device="mps",
        screenScaling=926,
        length_threshold=0,
        maxGameLength=25,
        prep_transform=None,
        augmentation_transform=None,
        label_transform=None,
        analysing_data=False,
    ):
        # screen scaling is hard coded at 926
        # this represents the largest height possible of an iPhone (model iPhone 14 Pro)
        # ths value is in 'points', a measurement defined by Swift/Apple as to have consistent measurements across iPhone models
        # maxGameLength (used for scaling) hardcoded at 25.

        if augmentation_transform is not None and prep_transform is None:
            raise ValueError(
                "Augmentation transform provided without a preparatory transform."
            )

        self.data_folder = data_folder
        self.device = device
        self.prep_transform = prep_transform
        self.prep_cache = (
            {}
        )  # prep transform can be expensive to compute, so cache to store results
        self.augmentation_transform = augmentation_transform
        self.label_transform = label_transform
        self.samples = []  # A list to store the data examples and labels
        self.analysing_data = analysing_data

        # Iterate through JSON files in the data folder
        for filename in os.listdir(data_folder):
            if filename.endswith(".json"):
                with open(os.path.join(data_folder, filename), "r") as json_file:
                    sample_data = json.load(json_file)

                    # poorly recorded data can have limited touch points
                    # threshold used to ignore samples with too few touch points recorded
                    if len(sample_data["touchData"]) > length_threshold:
                        # note using the same scaling for x and y values as to have only one scale for distance
                        screen_dim = {
                            "width": sample_data["screenSize"]["width"],
                            "height": sample_data["screenSize"]["height"],
                        }

                        # This will output scaled touch data
                        unpacked_touchData = UnpackTouchData(
                            sample_data["touchData"],
                            start_time=sample_data["randomPath"][0]["time"],
                            screen_dim=screen_dim,
                            scale=screenScaling,
                            game_length=maxGameLength,
                        )

                        # Random path data is recorded as scaled, but from -0.5 to 0.5
                        # Hence, shift = 0.5 to give normalised coordinates from 0 to 1
                        unpacked_randomPath = UnpackPathData(
                            sample_data["randomPath"],
                            scale=screenScaling,
                            screen_dim=screen_dim,
                            start_time=sample_data["randomPath"][0]["time"],
                            shift=0.5,
                            game_length=maxGameLength,
                        )

                        label = sample_data[label_name]

                        sample = {
                            "touchData": unpacked_touchData,
                            "randomPath": unpacked_randomPath,
                            "screenHeight": sample_data["screenSize"]["height"],
                            "controlPoints": sample_data["controlPoints"],
                            "animationDuration": sample_data["animationDuration"],
                            "gameLength": maxGameLength,
                            "id": sample_data["id"],
                        }

                        self.samples.append((sample, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx in self.prep_cache:  # if already applied prep transform
            sample_data = self.prep_cache[idx]
        elif self.prep_transform:  # if haven't applied prep transform, and it exists
            sample_data, _ = self.samples[idx]
            sample_data = self.prep_transform(sample_data)
            self.prep_cache[idx] = sample_data
        else:  # if no prep transform
            sample_data, _ = self.samples[idx]

        _, label = self.samples[idx]

        if self.augmentation_transform:
            sample_data = self.augmentation_transform(sample_data)
            sample_data = sample_data.to(self.device)

        if self.label_transform:
            label = self.label_transform(label)
            label = label.to(self.device)

        if self.analysing_data:  # for when carrying out statistics to analyse data
            return sample_data, label, self.samples[idx][0], self.samples[idx][0]["id"]
        else:
            return sample_data, label


def UnpackTouchData(data, start_time, scale=None, game_length=1, screen_dim=None):
    # necessary to centre normalisation around 0.5
    y_shift = (scale - screen_dim["height"]) / 2
    x_shift = (scale - screen_dim["width"]) / 2

    x = [(point["xLocation"] + x_shift) / scale for point in data]
    y = [(point["yLocation"] + y_shift) / scale for point in data]

    time = [(point["time"] - start_time) / game_length for point in data]

    unpacked_data = {"X": x, "Y": y, "time": time}

    return unpacked_data


def UnpackPathData(
    data, start_time, scale=None, screen_dim=None, shift=0, game_length=1
):
    # raw X and Y is scaled from -0.5 to 0.5, representing percentage of that specific device's screen
    # so scaling needs to account for this

    x_factor = screen_dim["width"] / scale
    y_factor = screen_dim["height"] / scale

    centering_yshift = (1 - y_factor) / 2
    centering_xshift = (1 - x_factor) / 2

    x = [
        (((point["xDimScale"] + shift) * x_factor) + centering_xshift) for point in data
    ]
    y = [
        (((point["yDimScale"] + shift) * y_factor) + centering_yshift) for point in data
    ]

    time = [(point["time"] - start_time) / game_length for point in data]

    unpacked_data = {"X": x, "Y": y, "time": time}

    return unpacked_data


# %%
