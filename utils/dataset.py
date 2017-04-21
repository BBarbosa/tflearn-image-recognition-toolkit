from __future__ import division, print_function, absolute_import

import tflearn
import sys,math,time,os
import numpy as np
from tflearn.data_utils import shuffle,featurewise_zero_center,featurewise_std_normalization
from tflearn.data_utils import build_image_dataset_from_dir                 # for windows
#from tflearn.data_utils import build_hdf5_image_dataset,image_preloader     # for linux
from PIL import Image

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
def load_dataset_windows(train_path,height=None,width=None,test=None,shuffle=False,validation=0):
    """ 
    Given a folder containing images separated by folders (classes) returns training and testing
    data, if specified.
    """
    classes = X = Y = Xtr = Ytr = Xte = Yte= None

    print("Loading dataset (from directory)...")
    if(width and height):
        X,Y = build_image_dataset_from_dir(train_path, resize=(width,height), convert_gray=False, 
                                           dataset_file=train_path, filetypes=None, shuffle_data=False, 
                                           categorical_Y=True)
    else:
        X,Y = build_image_dataset_from_dir(train_path, resize=None, convert_gray=False, dataset_file=train_path, 
                                           filetypes=None, shuffle_data=False, categorical_Y=True)
    
    width,height,ch = X[0].shape            # get images dimensions
    nimages,classes = Y.shape               # get number of images and classes    

    #------------------------------ validation split ------------------------------------------------
    if(validation > 0 and validation < 1):  # validation = [0,1] float
        counts  = [0] * classes             # create an array to store the number of images per class

        for i in range(0,nimages):
            counts[np.argmax(Y[i])] += 1    # counts the number of images per every class
        
        print("\t       Images: ", nimages)
        print("\t       Counts: ", counts)
        print("\t      Classes: ", classes)

        Xtr = []
        Xte = []
        Ytr = []
        Yte = []

        # split train and test data manually, according to the value of the validation set
        it = 0
        for i in range(0,classes):
            it = 0 
            it += sum(counts[j] for j in range(0,i))
            
            per_class = counts[i]                           # gets the number of images per class
            to_test   = math.ceil(validation * per_class)   # calculates how many images to test per class
            split     = it + per_class - to_test            # calculates the index that splits data in train/test
            
            if False: print("%4d %4d %4d" % (it,split,split+to_test))

            Xtr += X[it:split]
            Ytr = np.concatenate([Ytr, Y[it:split]]) if len(Ytr)>0 else Y[it:split] 
        
            #print(Ytr[i*per_class:i*per_class+split])
            #inp = input("Press any key...")
            
            Xte += X[split:split+to_test]
            Yte = np.concatenate([Yte, Y[split:split+to_test]]) if len(Yte)>0 else Y[split:split+to_test]
        
            #print(Yte[i*45:i*45+to_test])
            #inp = input("Press any key...")

    #----------------------------------------------------------------------------------------------
    else:
        Xtr = X
        Ytr = Y
    
    del(X)
    del(Y)
    
    Xtr = np.array(Xtr)     # convert train images list to array
    Ytr = np.array(Ytr)     # convert train labels list to array
    Xte = np.array(Xte)     # convert test images list to array
    Yte = np.array(Yte)     # convert test labels list to array

    if(shuffle):
        Xtr,Ytr = shuffle(Xtr,Ytr)
        Xte,Yte = shuffle(Xte,Yte)

    Xtr = np.reshape(Xtr,(-1,height,width,ch))      # reshape array to fit on network format
    Xte = np.reshape(Xte,(-1,height,width,ch))      # reshape array to fit on network format

    print("\t         Path: ",train_path)
    print("\tShape (train): ",Xtr.shape,Ytr.shape)
    if(validation > 0):
        # only prints if they are allocated
        print("\tShape  (test): ",Xte.shape,Yte.shape)
    print("Data loaded!\n")

    return classes,Xtr,Ytr,height,width,ch,Xte,Yte

# function that loads a set of test images
def load_test_images(testdir=None):
    image_list = []
    label_list = []
    classid = -1
    
    if(testdir):
        print("Loading test images...")
        # picks every sa
        for root, dirs, files in os.walk(testdir):
            for file in files:
                if file.endswith((".bmp",".jpg")):
                    image_path = os.path.join(root, file)
                    image      = Image.open(image_path)
                    image_list.append(image)
                    label_list.append(classid)

            classid += 1
        
        print("\t       Images: ",len(image_list))
        print("\t       Labels: ",len(label_list))
        print("Test images loaded...\n")
    else:
        print("WARNING: Path to test image is not set")
    
    return image_list,label_list