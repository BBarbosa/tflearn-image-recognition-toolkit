from __future__ import division, print_function, absolute_import

import sys, os, platform, time
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
from utils import architectures, dataset
from colorama import init
from termcolor import colored

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# init colored print
init()

# initial check. exits if there isn't enough arguments --------------------------------------------------
if (len(sys.argv) < 5):
    print(colored("Call: $ python evolution.py {dataset} {architecture} {models_dir} {test_dir}","red"))
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

# loads train and validation sets
CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xt,Yt = dataset.load_dataset_windows(data,HEIGHT,WIDTH,shuffle=False,validation=0.1)

# NOTE: needs to load test images 


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

epoch = 0
array = []

train_acc = np.array(array, dtype = np.float32)
val_acc   = np.array(array, dtype = np.float32)
epochs    = np.array(array, dtype = np.int8)

# picks every saved model
for root, dirs, files in os.walk(modelsdir):
    for file in files:
        if file.endswith((".data-00000-of-00001")):
            modelpath = os.path.join(root, file)
            modelpath = modelpath.split(".")[0]

            print("-----------------------------")
            print("Loading trained model...")  
            model.load(modelpath)
            print("\tModel: ",modelpath)
            print("Trained model loaded!\n")

            stime = time.time()  
            
            train = model.evaluate(X,Y)[0]
            val   = model.evaluate(Xt,Yt)[0]

            train_acc = np.append(train_acc,train)
            val_acc   = np.append(val_acc,val)
            epochs    = np.append(epochs,epoch)
                
            print("  Training set:",train)
            print("Validation set:",val)
            
            ftime = time.time() - stime
            print("          Time: %.3f\n" % ftime)

            epoch += 1

# plot properties 
# fig = plt.figure()
plt.title("Impact of epochs number")
ptrain = plt.plot(epochs,train_acc,color='red')
ptest  = plt.plot(epochs,val_acc,color='blue')

red_patch   = mpatches.Patch(color='red', label='Train')
blue_patch  = mpatches.Patch(color='blue', label='Validation')
green_patch = mpatches.Patch(color='green', label='Test')

legend = plt.legend(handles=[red_patch,blue_patch],loc=7)
# location docs: http://matplotlib.org/1.3.1/users/legend_guide.html 

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.xlim(0,epoch-1)
plt.xticks(epochs)

plt.grid(True)
plt.show()  