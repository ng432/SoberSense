
## sobersensetools

Sobersensetools is a custom package used to prepare and model data from the SoberSense app. It includes custom data transforms for feature engineering and data augmentation, a custom dataclass and and basic neural networks. A data sample for a single recording has the following data structure:

[num_features, N]

**num_features &ge; 5, and include x_touch, x_path, y_touch, y_path, time...**

A slice i ([:,i]) of a recording represents the coordinates of touch, the coordinates of the circle and time at a single instance, as well as any additional engineered features. Other features that can be appended include cartesian distance between touch point and circle, velocity and acceleration of touch, and extracted reaction times. 


**N = number of touch points for a recording.**

The position of the circle is interpolated for the given time of each touch point. No zero padding. For more information on how N varies, see pilot_data_exploration.ipynb


## pilot_data_exploration.ipynb

This is a Jupyter notebook containing basic statistical analyses of the data, including reaction time analysis and pair plots of engineered features. It also includes the code used to derive 'av_normalising_values.json', containing the values used in touch acceleration and velocity normalisation. 

## running_models.py

This is the script used to train the neural networks found in sobersensetools, using TensorBoard to track loss changes. 


