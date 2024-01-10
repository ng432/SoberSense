
Code in Python and PyTorch to train NN models on recorded data from the app.

A data sample contains (an estimate of) the users BAC, touch data and path data.
Path data refers to the coordinates and timestamp of a circle's path as it (semi) randomly moves across the screen.
Touch data refers to the coordianates and timestamp of the users touch data as they try to keep their finger on the circle.
For a demonstration of what is meant by this, see root directory. 

**'models_and_transforms'** contains models and accomapnying data transforms for a sample data structure of:

[num_features, N]

num_features = at least 5: x_touch, x_path, y_touch, y_path, time
Extra features can be added include cartesian distance between touch point and circle, velocity and acceleration of touch, and reaction times.

N = number of touch points for a recording. The position of the circle is interpolated for the given time of each touch point. No zero padding.


