
# %%

import numpy as np

""" 
Functions written for Bezier interpolation of animation path of circle.
This is possible as the circle is animated with a Bezier time path with known control points.
For a Bezier timingCurve, the x axis represents progression of time (from 0 to 1) and the y represents progression in space (from 0 to 1).
"""


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
def AnimationPathBezierInterpolation(randomPath, number_of_points, control_points, animationDuration, game_length=1):
    # Number of points, is how many points are interpolated for EACH animation jump, between the start and end coordinate of a given jump
    # Currently returns interpolation for just duration of animation
    # Interpolation doesn't account for a static period (~0.4 seconds, or 0.02 in normalised time) between each jumps

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

# finding the parametric t value for a given x value
# x axis, in the context of a timing path, represents time 
def findParametricTforX(time, p0, p1, p2, p3, tolerance=1e-5):

    low, high = 0, 1

    while low < high:

        mid = (low + high) / 2

        if abs(BezierSingleCalculation(mid, p0, p1, p2, p3) - time) < tolerance:
            return mid
        if BezierSingleCalculation(mid, p0, p1, p2, p3) < time:
            low = mid
        else:
            high = mid
            
    return (low + high) / 2
# %%
