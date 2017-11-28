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
1. Download and install Python 3.5.x x64 version (confirm Pip option).
   [Python](https://www.python.org/downloads/release/python-352/)

2. (For GPU installation) Download and install Nvidia CUDA Toolkit.  
   See if graphics card is compatible on the link below. 
   [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

3. (For GPU installation) Download CuDNN. 
   See if graphics card is compatible. 
   There is also a backup copy of CuDNN on the installation instructions folder.
   [Nvidia cuDNN](https://developer.nvidia.com/cudnn)

   NOTE: The CuDNN version vary according to the TensorFlow version
    - TensorFlow version <= 1.2.1 -> CuDNN v5.1
    - TensorFlow version >  1.3.0 -> CuDNN v6

4. (For GPU installation) Copy CuDNN files to Nvidia CUDA toolkit folder after point 2.
   By default, it is located on C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0

   4.1 copy cudnn\bin\cudnn64_5.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\ (for CuDNN 5.1)
   4.2 copy cudnn\bin\cudnn64_6.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\ (for CuDNN 6)
   4.3 copy cudnn\include\cudnn.h   to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\
   4.4 copy cudnn\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\

5. Install Tensorflow via pip (on the commands line)
   pip install --upgrade tensorflow-gpu (for the latest stable GPU version) OR
   pip install tensorflow-gpu==1.1.0 (for a specific GPU version)           OR
   pip install tensorflow (for the latest CPU version)                      OR
   pip install tensorflow==1.1.0 (for a specific CPU version)               OR 
   Download from this link [Python's libs](http://www.lfd.uci.edu/~gohlke/pythonlibs/#tensorflow)

   NOTE: The CuDNN version vary according to the TensorFlow version
    - TensorFlow version <= 1.2.1 -> CuDNN v5.1
    - TensorFlow version >  1.3.0 -> CuDNN v6

   5.1 TensorFlow dependencies
      5.1.1 Make sure that Visual C++ Redistributate 2015 (or 2017) x64 is installed. 
            If not, download it from the link underneath .
            [MS Visual C++ Redistributate 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)
       
      5.1.2 Move _curses_curses.cp35-win_amd64.pyd and _curses_panel.cp35-win_amd64.pyd to
            Python's installation folder.
            By default, it is on
            C:\Users\<Username>\AppData\Local\Programs\Python\Python35\Lib\site-packages
            Unless the user set it to other location
       
      5.1.3 Download and install these packages via pip
            Move to the directory where the packages were downloaded, open a command prompt 
            and make 'pip install <package_name>'
         
         5.1.3.1 [Numpy+MKL](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
            - download numpy‑1.13.10+mkl‑cp35‑cp35m‑win_amd64.whl
            - pip install numpy‑1.13.0+mkl‑cp35‑cp35m‑win_amd64.whl
            - NOTE: Also available on the installation folder 
         
         5.1.3.2 [Scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)
            - download scipy‑0.19.0‑cp35‑cp35m‑win_amd64.whl
            - pip install scipy‑0.19.0‑cp35‑cp35m‑win_amd64.whl
            - NOTE: Also available on the installation folder 

6. Install TFLearn via pip (on the commands line)
   pip install tflearn

7. Additional code dependencies
   In one single step by
   - pip install -r requirements.txt
   
   Or manually by
   - pip install opencv-python
   - pip install colorama
   - pip install termcolor 
   - pip install h5py

8. Test installation
   Open a command line on the installation folder and run
   - 'python training_template.py -h' for getting a help hint
   - 'python training_template.py' to train a CNN with MNIST database
   
