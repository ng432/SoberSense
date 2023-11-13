
Code in Python and PyTorch to train NN models on recorded data from the app.

A data sample contains (an estimate of) the users BAC, touch data and path data.
Path data refers to the coordinates and timestamp of a circle's path as it (semi) randomly moves across the screen.
Touch data refers to the coordiantes and timestamp of the users touch data as they try to keep their finger on the circle.

**'models_and_transforms_v1'** contains models and accompanying data transforms for a sample data structure of:
[Ch, D, N]

Ch = 2, one dimension each for touch data and path data

D = 3, for x y and time stamp of the point

This data structure interpolates the Bezier based animation path of the circle by a given number of points between the start and end of each movement of the circle to give N data points. The touch data is then zero-padded to match N.


