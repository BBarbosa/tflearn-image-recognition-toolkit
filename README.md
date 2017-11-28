# Deep Learning Image Classifier Toolkit
Deep learning image classifier toolkit supported by **TFLearn** on top of **Tensorflow**. 

## Folders descriptions
* scripts/           - auxiliary scripts to preprocessing data and/or for data agumentation
* utils/             - built-in modules imported on the code
  * architectures.py - module where are defined all networks architectures
  * classifier.py    - module that contains a set of image classification methods
  * communicator.py  - module to communicate with a local server through sockets (only sends)
  * dataset.py       - module responsible for loading data from folder / .hdf5 / .pkl
  * helper.py        - module with a bunch of helper functions
* classify.py        - script that runs a trained model 
* training.py        - script that trains a model
* dataset/           - directory contains raw images ready to be cropped so they fit on CNNs
  * dataset-name/    - designation for a specific dataset
     * training/     - sub-directory containing images used for training
        * class-0    
        * class-1    
        * ...
        * class-N 
     * testing/      - sub-directory that contains testing images
        * class-0    
        * class-1    
        * ...
        * class-N

## Windows installation
1. Download and install Python 3.5.x x64 version (confirm Pip option).
   [Python's Installer](https://www.python.org/downloads/release/python-352/)

2. (For GPU installation) Download and install Nvidia CUDA Toolkit.  
   Check if graphics card is compatible on the link below. 
   [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

3. (For GPU installation) Download CuDNN. 
   Check if graphics card is compatible. 
   [Nvidia cuDNN](https://developer.nvidia.com/cudnn)

   **Note:** The CuDNN version vary according to the TensorFlow version
   - TensorFlow version <= 1.2.1 -> CuDNN v5.1
   - TensorFlow version >  1.3.0 -> CuDNN v6

4. (For GPU installation) Copy CuDNN files to Nvidia CUDA toolkit folder after finished point 2.
   By default, it is located on C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0

   - copy _cudnn\bin\cudnn64_5.dll_ to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\ (for **cuDNN 5.1**)
   - copy _cudnn\bin\cudnn64_6.dll_ to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\ (for **cuDNN 6**)
   - copy _cudnn\include\cudnn.h_   to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\
   - copy _cudnn\lib\x64\cudnn.lib_ to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\

5. Install [Tensorflow](https://github.com/tensorflow/tensorflow)
   - ```pip install --upgrade tensorflow-gpu``` (for the latest stable GPU version) **OR**
   - ```pip install tensorflow-gpu==1.1.0``` (for a specific GPU version)           **OR**
   - ```pip install tensorflow``` (for the latest CPU version)                      **OR**
   - ```pip install tensorflow==1.1.0``` (for a specific CPU version)               **OR** 
   - Direct download from [Windows Python's Libs](http://www.lfd.uci.edu/~gohlke/pythonlibs/#tensorflow)

   **Note:** The CuDNN version vary according to the TensorFlow version
   - TensorFlow version <= 1.2.1 -> CuDNN v5.1
   - TensorFlow version >  1.3.0 -> CuDNN v6

   **TensorFlow dependencies**
   
   5.1 Make sure that Visual C++ Redistributable 2015 (or 2017) x64 is installed. 
       [MS Visual C++ Redistributable 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)
       
   5.2 Download and install these packages via pip. Move to the directory where the packages were downloaded. Open a command prompt 
       and make `pip install <package_name>`

   - [Curses](https://www.lfd.uci.edu/~gohlke/pythonlibs/#curses)
      - download curses‑2.2‑cp35‑none‑win_amd64.whl
      - ```pip install curses‑2.2‑cp35‑none‑win_amd64.whl```

   - [Numpy+MKL](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
      - download numpy‑1.13.10+mkl‑cp35‑cp35m‑win_amd64.whl
      - ```pip install numpy‑1.13.0+mkl‑cp35‑cp35m‑win_amd64.whl```
         
   - [Scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)
      - download scipy‑0.19.0‑cp35‑cp35m‑win_amd64.whl
      - ```pip install scipy‑0.19.0‑cp35‑cp35m‑win_amd64.whl```

6. Install [TFLearn](https://github.com/tflearn/tflearn)
   - ```pip install tflearn```

7. Additional code dependencies
   
   In one single step by
      - ```pip install -r requirements.txt```
   
   Or manually by
      - ```pip install opencv-python```
      - ```pip install colorama```
      - ```pip install termcolor```
      - ```pip install h5py```

8. To test the installation, open a command line on the installation folder and run
   
   ```python -c "import tensorflow as tf; print(tf.__version__)" ```
