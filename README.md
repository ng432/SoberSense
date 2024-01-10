# SoberSense
... is a personal project aiming to predict blood alcohol concentration (BAC) from touch data using a neural network.

## SoberSenseApp
This directory contains the Swift code for an iOS app designed to record a user's touch data during a simple game, and collect data such as how much alcohol the user has drunk at the time of recording. The game consists of a circle moving semi-randomly across the screen, while the aim is to keep the index finger of the dominant hand on the circle.

**Demo of app:**

https://github.com/ng432/SoberSense/assets/73446355/45c619fc-ed20-47b0-80d2-d2a83e8ee1df


## Data_analysis 
This directory contains Python code to model the collected data using neural networks.

### Framing of task
Currently, the task is framed as a binary classification, either above or below the British drink driving limit (BAC = 0.08), from time-series touch data. The BAC of an individual for a given recorded game is estimated from their weight, sex, how much they have drunk, and how long ago they had their first drink using the Widmark formula, and it is then assessed as to whether it is above or below 0.08 to give a label for a given data sample. 

### Baseline comparison
In order to relate the efficacy of a neural network, it is necessary to have a baseline comparison. For this sake, a simple classifier was created by fitting a linear regression to average reaction times derived from a sample derived, and then assessing whether, on average, a sample's reaction times were above or below the reaction time threshold for being above the limit. This gives a **precision of 0.55, recall of 0.65, and F1 Score of 0.60.**

### Top performance 
Current top-performance is with a **convolutional neural network** applied to a randomly cropped continuous segment of the touch data - this is repeated multiple times so that for a given sample all data points are used, and then **majority voting** across the repetitions decides whether the sample is predicted as above or below the limit. This gives a **precision of 0.60, recall of 0.74, and F1 Score of 0.67.**










