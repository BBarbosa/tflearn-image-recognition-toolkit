# Image classifier
This repository will host my experiments that will be used as support to my MSc Thesis

## Folders descriptions
* scripts - auxiliary scripts to preprocessing data 
* utils   - built modules imported on the code
  * architectures.py - file where are defined all networks architectures
  * dataset.py       - file responsible for loading data from folder / .hdf5 / .pkl
* classify.py    - script that runs a trained model and classifies a single image
* classify_sw.py - script that runs a trained model and classifies a single image by sliding window
* training.py    - script that trains a model