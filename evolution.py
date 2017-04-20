from __future__ import division, print_function, absolute_import

import sys,os,platform,time,re,glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflearn
import tensorflow as tf
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected,flatten
from tflearn.data_utils import shuffle,featurewise_zero_center,featurewise_std_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tflearn.helpers.summarizer as s

import winsound as ws
import numpy as np
from utils import architectures,dataset,classifier
from colorama import init
from termcolor import colored

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# init colored print
init()

numbers = re.compile(r'(\d+)')      # regex for get numbers

def numericalSort(value):
    """
    Splits out any digits in a filename, turns it into an actual 
    number, and returns the result for sorting. Code from
    http://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def plot_accuracies(x,y):
    """
    Function that creates a plot according to X and Y. Lengths must match.

    Params:
        x - array for x-axis [0,1,2]
        y - assuming that it is a matrix so it can automatically generate legend [[3,4,5], [6,7,8], [9,0,1]]
    """
    # fig = plt.figure()
    plt.title("Impact of epochs number",fontweight='bold')

    legend = np.arange(0,len(y[0])).astype('str')      # creates array ['0','1','2',...,'n']
    plt.plot(x,y)
    plt.legend(legend) 

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x)

    plt.grid(True)
    plt.show() 

def parse_error_files(files_dir):
    """
    Function to parse srtructured error files.

    Params:
        files_dir - directory where error files are stored
    """
    
    for infile in sorted(glob.glob(files_dir + '*.txt'), key=numericalSort):
        print("File: " + infile)


# -------------------------------------------------------------------------------------------------------
if(len(sys.argv) == 2):
    parse_error_files(sys.argv[1])
    #plot_accuracies([1,2],[[1,2,3],[4,5,6]])
    sys.exit(1)
elif (len(sys.argv) < 5):
    print(colored("Call: $ python evolution.py {dataset} {architecture} {models_dir} {test_dir} \t OR","red"))
    print(colored("Call: $ python evolution.py {error_dir} ","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))
# -------------------------------------------------------------------------------------------------------

# specify OS
OS = platform.system() 

# clear screen and show OS
if(OS == 'Windows'):
    os.system('cls')
else:
    os.system('clear')
print("Operating System: %s\n" % OS)

# change if you want a specific size
HEIGHT = None
WIDTH  = None

# get command line arguments
data      = sys.argv[1]       # path to hdf5/file.pkl OR path/to/cropped/images
arch      = sys.argv[2]       # name of architecture
modelsdir = sys.argv[3]       # path for trained model
testdir   = sys.argv[4]       # path for test directory

# bunch of control flags
use_train = True
use_test  = True
per_class = True

# loads train and validation sets
if(use_train):
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv = dataset.load_dataset_windows(data,HEIGHT,WIDTH,shuffle=False,validation=0.1)

# load test images 
if(use_test):
    Xt,Yt = dataset.load_test_images(testdir)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_samplewise_zero_center()   # per sample (featurewise is a global value)
img_prep.add_samplewise_stdnorm()       # per sample (featurewise is a global value)
    
# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=45.)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS],    # shape=[None,IMAGE, IMAGE] for RNN
                     data_preprocessing=img_prep,       
                     data_augmentation=None) 

network = architectures.build_network(arch,network,CLASSES)
    
# model definition
model = tflearn.DNN(network, checkpoint_path=None, tensorboard_dir='logs/',
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95,
                    best_checkpoint_path=None)

epoch = 1
array = []
accuracies = []

#test_acc  = np.array(array, dtype = np.float32)
epochs    = np.array(array, dtype = np.int8)

# picks every saved model
for infile in sorted(glob.glob(modelsdir + '*.data-00000-of-00001'), key=numericalSort):
    modelpath = infile.split(".")[0]
    
    print("----------------------------------")
    print("Loading trained model...")  
    model.load(modelpath)
    print("\tModel: ",modelpath)
    print("Trained model loaded!\n")
    
    print("EPOCH: ",epoch)
    stime = time.time()  
    accuracy,_,_,_ = classifier.classify_sliding_window(model,Xt,Yt,epoch,CLASSES)
    ftime = time.time() - stime
    
    print("Time: %.3f\n" % ftime)
    epochs     = np.append(epochs,epoch)                                                # epochs: [1,2,3,...,n]
    accuracies = np.vstack((accuracies,accuracy)) if len(accuracies) > 0 else accuracy  # acc: [[98,65,64,23,...],[43,54,65,87,...],...]
    epoch += 1

plot_accuracies(epochs,accuracies)