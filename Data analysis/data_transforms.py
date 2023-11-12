

#%% 

import torch as t
from math import sqrt
from bezier_interpolation import AnimationPathBezierInterpolation

# Contains transforms for data pre-processing and data augmentation

def full_augmentation_transform(sample_data, variance = 0.02, new_length = 0.2):

    sample_data = prep_transform(sample_data)

    # cropping data
    sample_data = croptime_sample(sample_data, new_length = new_length)

    # flipping data
    sample_data = randomly_flipx(sample_data)
    sample_data = randomly_flipy(sample_data)

    # adding noise
    sample_data = addxynoise_totouch(sample_data, variance = variance)

    return sample_data 

    

def prep_transform(sample_data, device='mps', animation_interp_number = 60):
    # Prepatory transform for collected data 
    # Returns tensor of shape:  [Ch, D, S]
    # Ch (=2): channels, one representing touch data, one representing animation path of circle
    # D (=3): x coordinate, y coordinate, time
    # S: Number of samples of animation path (including those interpolated). touchData padded with 0s to match animation path length

    # interpolating animation on first access of data
    if "interpolated" not in sample_data:

        sample_data["randomPath"] = AnimationPathBezierInterpolation(
                            sample_data["randomPath"],
                            animation_interp_number,
                            sample_data["controlPoints"],
                            sample_data["animationDuration"],
                            game_length=sample_data["gameLength"],
                        )
        
        sample_data["interpolated"] = True

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


def randomly_flipx(sample):

    num = t.rand(1).item()

    if num > 0.5:
        sample[...,0,:] *= -1
        
        # necessary to insure zero padding isn't flipped to 1 
        non_zero_mask = (sample[0,0,:] != 0)

        sample[...,0,non_zero_mask] += 1 

    return sample

def randomly_flipy(sample):

    num = t.rand(1).item()

    if num > 0.5:
        sample[...,1,:] *= -1

        # necessary to insure zero padding isn't flipped to 1 
        non_zero_mask = (sample[0,1,:] != 0)

        sample[...,1,non_zero_mask] += 1 

    return sample

def addxynoise_totouch(sample, variance = 0.02):

    # need to avoid adding noise to zero padding  
    noise_mask = (sample[0,0,:] != 0)

    noise_shape = sample[0, :2, noise_mask].shape 

    noise = t.randn(noise_shape)*sqrt(variance)
    
    # TODO finish sorting device situation

    # noise.to(device)

    sample[0, :2, noise_mask] += noise

    # ensuring that noise doesn't give erroneous values
    sample = t.clamp(sample, min = 0, max = 1)

    return sample

# crops batched data with frame size of new_length relevant to the last dimension
# new_length is from 0 to 1 and gives scale of cropping
# e.g. new_length = 0.1 data cropped to 10% of full length
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
def croptime_sample(sample, new_length = 0.2):

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

    # TODO: add error handling for bad padding to see issue 

    padded_td = t.nn.functional.pad(touch_data, (front_pad, back_pad))

    cropped_sample = t.stack((padded_td, path_data))

    return cropped_sample




# %%

