from __future__ import division, print_function, absolute_import

import tflearn
#import h5py # used on linux only
import sys
import numpy as np
from tflearn.data_utils import shuffle,featurewise_zero_center,featurewise_std_normalization
from tflearn.data_utils import build_image_dataset_from_dir                 # for windows
#from tflearn.data_utils import build_hdf5_image_dataset,image_preloader     # for linux

# create dataset for HDF5 format
def create_dataset(train_path,height,width,output_path,test_path=None,mode='folder'): 
    train_output_path = test_output_path = None

    print("Creating dataset...")
    if (train_path):
        train_output_path = "%s%s" % (output_path,"_train.h5")
        #build_hdf5_image_dataset(train_path, image_shape=(height,width), mode=mode, output_path=train_output_path, categorical_labels=True, normalize=True, grayscale=False)
        print("\tTrain: ",train_output_path)
    else:
        sys.exit("ERROR: Path to train dataset not set!")
    
    if(test_path):
        test_output_path = "%s%s" % (output_path,"_test.h5")
        #build_hdf5_image_dataset(test_path, image_shape=(height,width), mode=mode, output_path=test_output_path, categorical_labels=True, normalize=True, grayscale=False)
        print("\t Test: ",test_output_path)

    print("Dataset created!\n")
    return train_output_path, test_output_path

# load dataset on format HDF5
def load_dataset(train,height,width,test=None):
    classes = X = Y = Xt = Yt = None

    print("Loading dataset (hdf5)...")
    if(train):
        h5f = h5py.File(train, 'r')
        X = h5f['X'] 
        X = np.array(X)                        # convert to numpy array
        X = np.reshape(X,(-1,height,width,3))  # reshape array to a suitable format
        #X = np.reshape(X,(-1,IMAGE,IMAGE))    # for RNN
        Y = h5f['Y']
        print("\tShape (train): ",X.shape,Y.shape)
        classes = Y.shape[1]
    else:
        sys.exit("ERROR: Path to train dataset not set!")
    
    if(test):
        h5f2 = h5py.File(test, 'r')
        Xt = h5f2['X'] 
        Xt = np.array(Xt)                        # convert to numpy array
        Xt = np.reshape(Xt,(-1,height,width,1))
        #Xt= np.reshape(X,(-1,height,width))     # for RNN  
        Yt = h5f2['Y']
        print("\tShape  (test): ",Xt.shape,Yt.shape)
    
    print("Data loaded!\n")
    return classes,X,Y,Xt,Yt

# image preload (alternative to HDF5)
def load_dataset_ipl(train_path,height,width,test_path=None,mode='folder'):
    classes = X = Y = Xt = Yt = None
    
    print("Loading dataset (image preloader)...")
    if(train_path):
        #X, Y = image_preloader(train_path, image_shape=(width,height), mode=mode, categorical_labels=True, normalize=True)
        classes = Y.shape[1]
    else:
        sys.exit("ERROR: Path to train dataset not set!")

    print("Data loaded!\n")    
    return classes,X,Y,Xt,Yt 

# load images directly from images folder (ex: cropped/5/)
def load_dataset_windows(train_path,height=None,width=None,test=None):
    classes = X = Y = None

    print("Loading dataset (from directory)...")
    if(width and height):
        X,Y = build_image_dataset_from_dir(train_path, resize=(width,height), convert_gray=False, 
                                           dataset_file=train_path, filetypes=None, shuffle_data=True, 
                                           categorical_Y=True)
    else:
        X,Y = build_image_dataset_from_dir(train_path, resize=None, convert_gray=False, dataset_file=train_path, 
                                           filetypes=None, shuffle_data=True, categorical_Y=True)

    # convert to array
    Y = np.array(Y)
    X = np.array(X)

    width,height,_ = X[0].shape
    X = np.reshape(X,(-1,height,width,3))
    classes = Y.shape[1]

    print("\t         Path: ",train_path)
    print("\tShape (train): ",X.shape,Y.shape)
    print("Data loaded!\n")

    return classes,X,Y,height,width

