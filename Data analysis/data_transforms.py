

#%% 

import torch as t
from math import sqrt

# Contains transforms for data pre-processing and data augmentation

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


def flipx(sample):
    sample[...,0,:] *= -1
    return sample

def flipy(sample):
    sample[...,1,:] *= -1
    return sample

def addxynoise(sample, variance):

    noise_tensor = t.zeros(sample.shape)
    noise_shape = noise_tensor[...,0,:2,:].shape
    # only x and y of touch data 
    noise_tensor[...,0,:2,:] = t.randn(noise_shape)*sqrt(variance)

    return sample + noise_tensor

# crops data with frame size of new_length relevant to the last dimension
# new_length is from 0 to 1 and gives scale of cropping
# e.g. new_length = 0.1 data cropped to 10% of full length
# crops based on the time of the path data
def croptime_batch(batch, new_length = 0.2):

    if new_length > 1 or new_length < 0:
        raise ValueError("new_length must be between 0 and 1")
    
    sample_list = []
    # running through samples in batch
    for i in range(batch.shape[0]):

        cropped_sample = croptime_sample(batch[i], new_length)
        
        sample_list.append(cropped_sample)

    return t.stack(sample_list, dim=0)


# takes a sample of [2, 3, N] (i.e. individual sample from a batch)
def croptime_sample(sample, new_length):

    num_points = sample.shape[-1]

    frame_size = int(new_length * num_points) 
    
    start_index = int(t.rand(1).item() * (num_points - frame_size))

    path_data = sample[1, :, start_index:start_index + frame_size]

    # taking the min and max times from path data, to then select touch data 
    min_time = path_data[2,0].item()
    max_time = path_data[2,-1].item()

    # mask of touch data samples that have the same time frame as cropped path data 
    min_mask = (sample[0,2,:] > min_time)

    # Necessary to deal with zero padding already added at end of tensor
    # Therefore when max_mask is taken, it won't always take index to end of tensor 
    sample[0,2, (sample[0,2,:] == 0)] = 1.1 

    max_mask = (sample[0,2,:] < max_time)

    # extracting the indices for the times in touch data that match the cropped path data
    touch_start_index = t.nonzero(min_mask)[0]
    touch_end_index = t.nonzero(max_mask)[-1]

    touch_data = sample[0,:, touch_start_index:touch_end_index]

    touch_length = touch_data.shape[-1] 

    if touch_length > frame_size:
        raise ValueError(f"Cropped touch data (length = {touch_length}) has more data points than path data (length = {frame_size}). Try more interpolation points for path data.")
    
    # padding touch data
    padding_length = frame_size - touch_length
    front_pad = padding_length // 2
    back_pad = padding_length - front_pad
    padded_td = t.nn.functional.pad(touch_data, (front_pad, back_pad))

    cropped_sample = t.stack((padded_td, path_data))

    return cropped_sample




# %%


# testing croptime_batch

# TODO: work out why it's not varying the seleceted data in this batch

f = data_set[0][0]

fake_batch = t.stack((f,f,f,f))

test = croptime_batch(fake_batch, new_length=0.1)


# %%
