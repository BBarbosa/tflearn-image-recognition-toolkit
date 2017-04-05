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

if (len(sys.argv) < 4):
    print(colored("Call: $ python training.py {dataset} {architecture} {output}","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))
else:
    # specify OS
    OS = platform.system() 
    
    # clear screen and show OS
    if(OS == 'Windows'):
        os.system('cls')
    else:
        os.system('clear')
    print("Operating System --> %s\n" % OS)

    # change if you want a specific size
    HEIGHT = None
    WIDTH  = None
    
    # get command line arguments
    data = sys.argv[1]  # path to hdf5/file.pkl OR path/to/cropped/images
    arch = sys.argv[2]  # name of architecture
    out  = sys.argv[3]  # name for output model

    vs = 0.1           # percentage of dataset for validation (manually)
    
    # load dataset and get image dimensions
    if(vs):
        CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xt,Yt = dataset.load_dataset_windows(data,HEIGHT,WIDTH,shuffle=False,validation=vs)
    else:
        CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,_,_= dataset.load_dataset_windows(data,HEIGHT,WIDTH,shuffle=False)
    
    
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
    network = input_data(shape=[None, WIDTH, HEIGHT, CHANNELS],    # shape=[None,IMAGE, IMAGE] for RNN
                        data_preprocessing=img_prep,       
                        data_augmentation=None) 
    
    network = architectures.build_network(arch,network,CLASSES)
        
    # model definition
    model = tflearn.DNN(network, checkpoint_path=out, tensorboard_dir='logs/', #session=sess,
                        max_checkpoints=1, tensorboard_verbose=0)  
    
    # training parameters
    bs    = 32                               # batch size [default=32]
    vs    = 0.1                              # percentage of dataset for validation
    dsize = X.shape[0]                       # size of dataset
    snap  = 50*(dsize - vs * dsize) // bs    # snapshot for each X times it passes through all data (integer division)     
    
    # callback definition for early stop 
    # NOTE: requires snapshoe_epoch=True
    class EarlyStoppingCallback(tflearn.callbacks.Callback):
        def __init__(self, acc_thresh):
            """
            Args:
                acc_thresh - if our accuracy > acc_thresh, terminate training.
            """
            self.acc_thresh = acc_thresh
            self.accs = []
        
        def on_epoch_end(self, training_state):
            """ """
            self.accs.append(training_state.global_acc)
            if training_state.val_acc is not None and training_state.val_acc > self.acc_thresh:
                raise StopIteration
    
    class MonitorCallback(tflearn.callbacks.Callback):
        def __init__(self, api):
            self.my_monitor_api = api

        def on_epoch_end(self, training_state):
            self.my_monitor_api.send({
                accuracy: training_state.global_acc,
                loss: training_state.global_loss,
            })

    # Initializae our callback.
    early_stopping_cb = EarlyStoppingCallback(acc_thresh=0.8)

    # training operation (default = 25epochs,10epochs 9 classes)
    model.fit(X, Y, n_epoch=200, shuffle=True, 
            show_metric=True, batch_size=bs, snapshot_step=snap,
            snapshot_epoch=False, run_id=out, validation_set=(Xt,Yt),
            callbacks=None)
    
    # save model
    modelname = "models/%s%s" % (out,".tflearn")
    model.save(modelname)

    if(OS == 'Windows'):
        freq = 1000
        dur  = 1500 
        ws.Beep(freq,dur)