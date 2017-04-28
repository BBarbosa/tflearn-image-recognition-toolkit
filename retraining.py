from __future__ import division, print_function, absolute_import
import sys, os, platform
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
from utils import architectures, dataset
from colorama import init
from termcolor import colored

# init colored print
init()

if (len(sys.argv) < 5):
    print(colored("Call: $ python retraining.py {dataset} {architecture} {batch_size} {runid} {model}","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))

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
data = sys.argv[1]          # path to hdf5/file.pkl OR path/to/cropped/images
arch = sys.argv[2]          # name of architecture
bs   = int(sys.argv[3])     # bacth size
out  = sys.argv[4]          # name for output model
modelpath = sys.argv[5]     # path to the already trained model

vs = 0.1           # percentage of dataset for validation (manually)

# load dataset and get image dimensions
if(vs and True):
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xt,Yt = dataset.load_dataset_windows(data,HEIGHT,WIDTH,shuffled=True,validation=vs)
else:
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,_,_= dataset.load_dataset_windows(data,HEIGHT,WIDTH,shuffled=True)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_samplewise_zero_center()   # per sample (featurewise is a global value)
img_prep.add_samplewise_stdnorm()       # per sample (featurewise is a global value)

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=45.)

# computational resources definition
tflearn.init_graph(num_cores=8,gpu_memory_fraction=0.9)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS],    # shape=[None,IMAGE, IMAGE] for RNN
                     data_preprocessing=img_prep,       
                     data_augmentation=img_aug) 

network = architectures.build_network(arch,network,CLASSES)
    
# model definition
model = tflearn.DNN(network, checkpoint_path="models/%s" % out, tensorboard_dir='logs/',
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95,
                    best_checkpoint_path=None)  

# training parameters
dsize = X.shape[0]              # size of dataset
snap  = 10*dsize // bs          # snapshot for each X times it passes through all data (integer division)  

print("Batch size:",bs)
print("  Val. set:",vs,"%")
print(" Data size:",dsize)
print("  Snapshot:",snap,"\n")

print("Loading trained model...")  
model.load(modelpath)
print("\tModel: ",modelpath)
print("Trained model loaded!\n")

# training operation 
model.fit(X, Y, n_epoch=10, shuffle=True, show_metric=True, 
          batch_size=bs, snapshot_step=None, snapshot_epoch=True, 
          run_id=out, validation_set=(Xt,Yt), callbacks=None)

# save model
modelname = "models/%s-tflearn" % out
model.save(modelname)

if(OS == 'Windows'):
    freq = 1000
    dur  = 1500 
    ws.Beep(freq,dur)