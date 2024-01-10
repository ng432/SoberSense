
Swift code for app designed to collect touch data during a task, to infer BAC (blood alcohol concentration).

**Starting screen**
On the intial screen, the user enters their data (sex, weight, units drunk, time since first drink).
This data is used to calculate the BAC, which will be used as the label for recorded touch data, used to train a neural network model.

![IMG_9331](https://github.com/ng432/SoberSense/assets/73446355/faab1ce3-b59b-4bc4-95b7-cb549b24a6d1)

**Animation screen**
The next screen displays the instructions for the task, where the user has to keep their finger on a circle which moves to (semi) randomly generated coordinates.
The path of the circle and the touch is recorded, to be used as the data for a neural network. 

![IMG_9332](https://github.com/ng432/SoberSense/assets/73446355/88d1572c-c13e-4b3f-b7fe-c37177e461dc)


**Final screen**
The final screen gives an estimate of the users BAC (calculated from the data entered on the starting screen). 
It also has an option to share the recorded touch and path data via email. 

![IMG_9333](https://github.com/ng432/SoberSense/assets/73446355/a6ea3623-261a-4271-8a5e-f8ffbba3d690)

