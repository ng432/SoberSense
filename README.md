# SoberSense
... is an application aiming to predict blood alcohol concentration (BAC) from touch data.

**SoberSenseApp** contains the Swift code for an app designed to collect user data, such as how much alchohol they have drunk, and their touch data during a simple game. This data is then used to train models. The game consists of a circle moving semi-randomly across the screen, while the aim is to keep the index finger of the dominant hand on the circle. See demo below. 

**Data_analysis** contains Python code to model the collected data using neural networks, with the aim of predicting BAC from the data. 

_Framing of task:_
Currently, the task is framed as a binary classification, either above or below the British drink driving limit (BAC = 0.08) from time-series touch data. The BAC of an individual for a given recorded game is estimated from their weight, sex, how much they have drunk, and when they had their first drink - this is assessed as to whether it is above or below 0.08 to give a label for a given data sample. 

_Top performance:_
Current top-performance is with a convolutional neural network applied to a randomly cropped continuous segment of the touch data - this is repeated multiple times so that for a given sample all data points are used, and then majority voting across the repetitions decides whether the sample is predicted as above or below the limit. This gives a precision of 0.60, recall of 0.74, and F1 Score of 0.67. 



**Demo of app:**

https://github.com/ng432/SoberSense/assets/73446355/766bf43b-d6ff-4da6-b853-17fbef57cffc







