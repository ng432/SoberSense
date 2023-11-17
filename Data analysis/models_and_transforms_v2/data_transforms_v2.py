


import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch as t
from math import sqrt
from bezier_interpolation import findParametricTforX, BezierSingleCalculation

#%%

def prep_transform(sample_data, device='mps'):
    """
    Prepatory transform for collected data 
    Returns tensor of shape:  [D, S]
    D (=5): x and y coordinates for touch and path data, and time stamp 
    S: Number of samples of touch data. X and y coordinate of circle is interpolated for each time of touch data """
    
    touch_data = t.tensor(
        [sample_data["touchData"]["X"], sample_data["touchData"]["Y"], sample_data["touchData"]["time"]]
    )

    path_data = t.tensor(
        [sample_data["randomPath"]["X"], sample_data["randomPath"]["Y"], sample_data["randomPath"]["time"]]
    )

    path_x_and_y = bezier_interp_timetospace(touch_data[2], path_data, sample_data["controlPoints"], sample_data["animationDuration"], sample_data["gameLength"])

    if path_x_and_y.shape[-1] != touch_data.shape[-1]:
        raise ValueError(f"The number of sampled touch points ({path_x_and_y.shape[-1]}) do not match the number of interpolated path points ({path_x_and_y.shape.shape[-1]})")

    sample_tensor =t.zeros((6, path_x_and_y.shape[-1]))

    # collecting x values 
    sample_tensor[0] = touch_data[0]
    sample_tensor[1] = path_x_and_y[0]

    # collecting y values
    sample_tensor[2] = touch_data[1]
    sample_tensor[3] = path_x_and_y[1]

    # calculating distance
    sample_tensor[4] = calculate_cartesian_distance(
        sample_tensor[0], sample_tensor[1], sample_tensor[2], sample_tensor[3])

    # collecting time
    sample_tensor[5] = touch_data[2]

    # MPS requires float32
    if device == "mps":
        sample_tensor = sample_tensor.to(t.float32)
        
    sample_tensor = sample_tensor.to(device)

    return sample_tensor

def calculate_cartesian_distance(x1, x2, y1, y2):    
    return t.sqrt(t.pow((x1-x2),2) + t.pow((y1-y2),2))

def bezier_interp_timetospace(touch_times, path_data, control_points, animation_duration, game_length):
    """ 
    Input:
    touch_times: a tensor of time points of touch (shape [N])
    path_data: tensor of path data containing the positions and times of the animating circle
    control_points, animation_length, game_length: details defining of the Bezier timing curve for animation
     
    Output:
    tensor of interpolated X and Y coordinates of circle position at times found in touch_times (shape [2, N])
    """

    path_selection_indices = find_relevant_jump(touch_times, path_data[2])

    # for a given touch time, how far is it after previous circle animation?
    times_after_animation_start = touch_times - path_data[2,path_selection_indices]

    # in 0 to 1 scaled time, what is the length of the animation 
    scaled_animation_duration = animation_duration / game_length

    interp_x = t.zeros_like(touch_times)
    interp_y = t.zeros_like(touch_times)

    for i, path_index in enumerate(path_selection_indices):
        
        if touch_times[i] < 0:
            # occasionally touch times before animation has started can be recorded
            # in this case, the position of the circle is just it's starting position

            interp_x[i] = path_data[0, 0]
            interp_y[i] = path_data[1, 0]

        elif times_after_animation_start[i] < scaled_animation_duration:
            # animation is ongoing 

            progress_into_animation = times_after_animation_start[i] / scaled_animation_duration
            parametric_t = findParametricTforX(progress_into_animation, 0, control_points[0], control_points[2], 1, tolerance=1e-5)

            # value from 0 to 1, representing how far along animation is in space
            animation_progression = BezierSingleCalculation(parametric_t, 0, control_points[1], control_points[3], 1)

            start_x = path_data[0, path_index]
            end_x = path_data[0, path_index + 1]
            interp_x[i] = start_x + (end_x - start_x) * animation_progression

            start_y = path_data[1, path_index]
            end_y = path_data[1, path_index + 1]
            interp_y[i] = start_y + (end_y - start_y) * animation_progression
            
        else:
            # animation has ended, so coordinate is the end point of that animation
            interp_x[i] = path_data[0, path_index + 1]
            interp_y[i] = path_data[1, path_index + 1]


    return t.stack((interp_x, interp_y))


def find_relevant_jump(touch_times, path_times):
    """ Outputs tensor (shape == touch_times.shape) of relevant indices into path_times for tensor of touch times. Relevant means the animation jump which was occuring at the time of a given touch.  """

    if (touch_times > path_times[-1]).any():
        raise ValueError("There is a touch time which occurs after animation has finished.")

    times_expanded = touch_times.unsqueeze(1)

    mask = times_expanded < path_times

    # minus 1 necessary as want the index for the animation jump before that touch point
    path_selection_indices = mask.long().argmax(dim=1) - 1 

    # occasionally touch data can be recorded before the animation has started 
    # these will give a 'path_selected_index' value of -1
    # these are set to 0, and are dealt with in bezier_interp_timetospace()
    path_selection_indices[path_selection_indices < 0] = 0

    return path_selection_indices




#%%

