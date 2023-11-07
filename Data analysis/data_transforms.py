

#%% 

import torch as t
from math import sqrt

# Contains transforms for data pre-processing and augmentation
# Transforms for data samples assume data is of structure [B, 2, 3, N]
# B: batch size
# 2: touch data, random path data
# 3: x, y coordinates (of screen), and timestamp
# N: number of data points recorded


def flipx(sample):
    sample[...,0,:] *= -1
    return sample

def flipy(sample):
    sample[...,1,:] *= -1
    return sample

def addxynoise(sample, variance):

    noise_tensor = t.zeros(sample.shape)
    noise_shape = noise_tensor[...,0,:2,:].shape
    noise_tensor[...,0,:2,:] = t.randn(noise_shape)*sqrt(variance)

    return sample + noise_tensor


def croplastdim(sample, new_size):
    start = t.randint(high=sample.size(-1) - new_size, size=(1,)).item()
    return sample[..., start:start + new_size]


# %%
