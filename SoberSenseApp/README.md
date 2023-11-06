
Swift code for app designed to collect touch data during a task, to infer BAC (blood alcohol concentration).

**Starting screen**
On the intial screen, the user enters their data (sex, weight, units drunk, time since first drink).
This data is used to calculate the BAC, which will be used as the regression target of a neural network model.

**Animation screen**
The next screen starts the task, where the user has to keep their finger on a circle which moves to (semi) randomly generated coordinates.
The path of the circle and the touch is recorded, to be used as the data for a neural network. 

**Final screen**
The final screen gives an estimate of the users BAC (calculated from the data entered on the starting screen). 
It also has an option to share the recorded touch and path data via email. 
