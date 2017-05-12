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
import winsound as ws
import numpy as np
from utils import architectures,dataset,classifier
from colorama import init
from termcolor import colored

# init colored print
init()

if (len(sys.argv) < 5):
    print(colored("Call: $ python training.py {dataset} {architecture} {batch_size} {runid} [testdir]","red"))
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
traindir = sys.argv[1]       # path to hdf5/file.pkl OR path/to/cropped/images
arch = sys.argv[2]           # name of architecture
bs = int(sys.argv[3])        # batch size
run_id = sys.argv[4]         # name for output model

try: 
    testdir = sys.argv[5]    # test images directory
except:
    testdir = None

vs = 0.1           # percentage of dataset for validation (manually)

# load dataset and get image dimensions
if(vs and True):
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv = dataset.load_dataset_windows(traindir,HEIGHT,WIDTH,shuffled=True,validation=vs)
    classifier.HEIGHT = HEIGHT
    classifier.WIDTH = WIDTH
    classifier.IMAGE = HEIGHT
    classifier.CHANNELS = CHANNELS
else:
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,_,_= dataset.load_dataset_windows(traindir,HEIGHT,WIDTH,shuffled=True)

# load test images
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

# computational resources definition (made changes on TFLearn's config.py)
tflearn.init_graph(num_cores=4,gpu_memory_fraction=0.4,allow_growth=True)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS],    # shape=[None,IMAGE, IMAGE] for RNN
                     data_preprocessing=img_prep,       
                     data_augmentation=img_aug) 

network = architectures.build_network(arch,network,CLASSES)

# model definition
model = tflearn.DNN(network, checkpoint_path="models/%s" % run_id, tensorboard_dir='logs/',
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95,
                    best_checkpoint_path=None)  

# some training parameters
EPOCHS = 100                    # total number of epochs 
SNAP = 5                        # snapshot at each SNAP epochs
iterations = EPOCHS // SNAP     # number of iterations 

print("Batch size:", bs)
print("Validation:", vs)
print("    Epochs:", EPOCHS)
print("  Snapshot:", SNAP, "\n")

# creates a new accuracies' .csv
csv_file = "%s_accuracies.txt" % run_id
fcsv = open(csv_file,"w+")
fcsv.write("train,validation,test,min\n") # header NOTE: Review when there isn't test dataset
fcsv.close()

# training operation 
for i in range(iterations):
    # show training progress
    train_acc = np.round(model.evaluate(X,Y)[0],2) * 100
    val_acc = np.round(model.evaluate(Xv,Yv)[0],2) * 100
    
    test_acc = 0
    if(testdir): 
        _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,run_id,CLASSES,printout=False)
    
    fcsv = open(csv_file,"a+")
    fcsv.write("%.2f,%.2f,%.2f,%.2f\n" % (train_acc,val_acc,test_acc,min_acc))
    fcsv.close()

    print("     Train:", train_acc, "%")
    print("Validation:", val_acc, "%")
    if(testdir): 
        print("      Test:", test_acc, "%")
        print("       Min:", min_acc, "%\n") 

    # stop criteria (does it make sense?)
    if(True and train_acc > 97.5 and val_acc > 97.5 and test_acc > 97.5):
        break

    # makes sucessive trainings until reaches stop criteria
    model.fit(X, Y, n_epoch=SNAP, shuffle=True, show_metric=True, 
              batch_size=bs, snapshot_step=False, snapshot_epoch=False, 
              run_id=run_id, validation_set=(Xv,Yv), callbacks=None)

fcsv.close()

# save model
modelname = "models/%s.tflearn" % run_id
model.save(modelname)

if(OS == 'Windows'):
    freq = 1000
    dur  = 1500 
    ws.Beep(freq,dur)   