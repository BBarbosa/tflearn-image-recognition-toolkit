# Image classifier
Image classifier toolkit supported by TFLearn on top of Tensorflow. 

## Folders descriptions
* scripts/ - auxiliary scripts to preprocessing data 
* utils/   - built modules imported on the code
  * architectures.py - file where are defined all networks architectures
  * dataset.py       - file responsible for loading data from folder / .hdf5 / .pkl
* classify.py - script that runs a trained model 
* training.py - script that trains a model
* dataset/ - contains raw images ready to be cropped so they fit on CNNs
  * train/ - folder containing images used for training
  * test/ - folder containing images for testing the network

## Windows installation
