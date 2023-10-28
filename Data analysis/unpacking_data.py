# %%

# Code to unpack data recorded from the SoberSense app

import json
import os
import torch as t
from torch.utils.data import Dataset
import numpy as np


class SoberSenseDataset(Dataset):
    def __init__(
        self, data_folder, device="mps", sample_transform=None, label_transform=None, animation_interp_number=10
    ):
        self.data_folder = data_folder
        self.device = device
        self.sample_transform = sample_transform
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

                    # for test_data, where data doesn't have 'duration', but it was 20 seconds
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
                        start_time=sample_data["randomPath"][0]["time"],
                        scale=scale,
                        game_length=game_length,
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

                    interpolated_randomPath = AnimationPathBezierInterpolation(
                        unpacked_randomPath,
                        animation_interp_number,
                        sample_data["controlPoints"],
                        sample_data["animationDuration"],
                        game_length=game_length,
                    )

                    label = sample_data["unitsDrunk"]
                    sample = {"touchData": unpacked_touchData, "randomPath": interpolated_randomPath}
                    self.samples.append((sample, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_data, label = self.samples[idx]

        if self.sample_transform:
            sample_data = self.sample_transform(sample_data)
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


# Bezier calculation for single parameter values and control points
def BezierSingleCalculation(t, p0, p1, p2, p3):
    out = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t**2) * p2 + (t**3) * p3
    return out


# Bezier calculation for a given number of equal spaced points from 0 to 1 (but not including 0 and 1)
def BezierMultCalculation(number_of_points, control_points):
    t_values = np.linspace(0, 1, number_of_points + 2)
    t_values = t_values[1:-1]

    # X coordinates represent time through animation (on scale from 0 to 1)
    x_values = [BezierSingleCalculation(t, 0, control_points[0], control_points[2], 1) for t in t_values]

    # Y coordinates represent progression through animation (on scale from 0 to 1)
    # In this instance, this represents movement of circle from one coordinate, to next
    y_values = [BezierSingleCalculation(t, 0, control_points[1], control_points[3], 1) for t in t_values]

    return x_values, y_values


# Scaling normalized Bezier values for interpolation of cooordinate during animation
def BezierCoordInterpolation(current_coord, next_coord, progress_values):
    distance = next_coord - current_coord
    interpolated_coord = [progress * distance + current_coord for progress in progress_values]

    return interpolated_coord


# This interpolates the X, Y and time values of the moving circle during it's animation, given the start and end position, and animation duration
# This is possible as the circle is animated with a Bezier time path with known control points
def AnimationPathBezierInterpolation(randomPath, number_of_points, control_points, animationDuration, game_length=1):
    # Number of points, is how many points are interpolated for EACH animation jump, between the start and end coordinate of a given jump

    # Gives un-scaled values to use for interpolation
    time_values, progress_values = BezierMultCalculation(number_of_points, control_points)

    interpolatedPath = {"X": [], "Y": [], "time": []}

    # interpolating for each jump
    for i in range(0, len(randomPath["X"]) - 1):
        current_x = randomPath["X"][i]
        next_x = randomPath["X"][i + 1]

        interpolated_x = BezierCoordInterpolation(current_x, next_x, progress_values)

        interpolatedPath["X"].append(current_x)
        interpolatedPath["X"].extend(interpolated_x)

        current_y = randomPath["Y"][i]
        next_y = randomPath["Y"][i + 1]

        interpolated_y = BezierCoordInterpolation(current_y, next_y, progress_values)

        interpolatedPath["Y"].append(current_y)
        interpolatedPath["Y"].extend(interpolated_y)

        start_time = randomPath["time"][i]

        # time_prop is normalised time along animation from 0 to 1,
        interpolated_times = [start_time + time_prop * (animationDuration / game_length) for time_prop in time_values]

        interpolatedPath["time"].append(start_time)
        interpolatedPath["time"].extend(interpolated_times)

    return interpolatedPath


# %%
