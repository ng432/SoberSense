
## sobersensetools

Sobersensetools is a custom package used to prepare and model data from the SoberSense app. It includes custom data transforms for feature engineering and data augmentation, a custom dataclass and and basic neural networks. A data sample for a single recording has the following data structure:

[num_features, N]

**num_features >= 5: x_touch, x_path, y_touch, y_path, time...**

A slice i ([:,i]) of a recording represents the coordinates of touch, the coordinates of the circle and time at a single instance, as well as any additional engineered features. Other features that can be appended include cartesian distance between touch point and circle, velocity and acceleration of touch, and extracted reaction times. 


**N = number of touch points for a recording.**
The position of the circle is interpolated for the given time of each touch point. No zero padding.


## pilot_data_exploration

This is a ju

Code in Python and PyTorch to train NN models on recorded data from the app.

A data sample contains (an estimate of) the users BAC, touch data and path data.
Path data refers to the coordinates and timestamp of a circle's path as it (semi) randomly moves across the screen.
Touch data refers to the coordianates and timestamp of the users touch data as they try to keep their finger on the circle.
For a demonstration of this, see root directory. Both touch data and path data for a given sample are inputted into the neural networks.

sobersensetools is a custom package, containing a custom dataclass, models and transforms for a sample data structure of:



