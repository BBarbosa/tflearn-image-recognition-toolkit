"""
Dataset loader
"""

from __future__ import division, print_function, absolute_import

import os
import re
import cv2
import sys
import PIL
import math
import time
import glob
import h5py
import random
import tflearn
import numpy as np

from tflearn.data_utils import shuffle, build_image_dataset_from_dir, image_preloader , build_hdf5_image_dataset        
from PIL import Image
from colorama import init
from termcolor import colored
import matplotlib.pyplot as plt

try:
    import scipy.ndimage
except Exception:
    print("Scipy not supported!")

# init colored print
init()

numbers = re.compile(r'(\d+)')      # regex for get numbers

def numericalSort(value):
    """
    Splits out any digits in a filename, turns it into an actual 
    number, and returns the result for sorting. Code from
    http://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python

    For directories [1, 2, 3, 10]
    From CMD it gets [1, 10, 2, 3]
    With numerical sort it gets [1, 2, 3, 10]
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# create dataset for HDF5 format
def create_dataset(train_path, height, width, output_path, test_path=None, mode='folder'): 
    train_output_path = test_output_path = None

    print("Creating dataset...")
    if (train_path):
        train_output_path = "%s%s" % (output_path, "_train.h5")
        #build_hdf5_image_dataset(train_path, image_shape=(height, width), mode=mode, output_path=train_output_path, categorical_labels=True, normalize=True, grayscale=False)
        print("\tTrain: ", train_output_path)
    else:
        sys.exit("[ERROR] Path to train dataset not set!")
    
    if(test_path):
        test_output_path = "%s%s" % (output_path, "_test.h5")
        #build_hdf5_image_dataset(test_path, image_shape=(height, width), mode=mode, output_path=test_output_path, categorical_labels=True, normalize=True, grayscale=False)
        print("\t Test: ", test_output_path)

    print("Dataset created!\n")
    return train_output_path, test_output_path

# load dataset on format HDF5
def load_dataset_hdf5(train, height, width, test=None):
    classes = X = Y = Xt = Yt = None

    print("Loading dataset (hdf5)...")
    if(train):
        h5f = h5py.File(train, 'r')
        X = h5f['X'] 
        X = np.array(X)                        # convert to numpy array
        X = np.reshape(X, (-1, height, width, 3))  # reshape array to a suitable format
        #X = np.reshape(X, (-1, IMAGE, IMAGE))    # for RNN
        Y = h5f['Y']
        print("\tShape (train): ", X.shape, Y.shape)
        classes = Y.shape[1]
    else:
        sys.exit("[ERROR] Path to train dataset not set!")
    
    if(test):
        h5f2 = h5py.File(test, 'r')
        Xt = h5f2['X'] 
        Xt = np.array(Xt)                        # convert to numpy array
        Xt = np.reshape(Xt, (-1, height, width, 1))
        #Xt= np.reshape(X, (-1, height, width))     # for RNN  
        Yt = h5f2['Y']
        print("\tShape  (test): ", Xt.shape, Yt.shape)
    
    print("Data loaded!\n")
    return classes, X, Y, Xt, Yt

# image preload (alternative to HDF5)
def load_dataset_ipl(train_path, height, width, test_path=None, mode='folder'):
    classes = X = Y = Xt = Yt = None
    
    print("Loading dataset (image preloader)...")
    if(train_path):
        #X, Y = image_preloader(train_path, image_shape=(width, height), mode=mode, categorical_labels=True, normalize=True)
        classes = Y.shape[1]
    else:
        sys.exit("[ERROR] Path to train dataset not set!")

    print("Data loaded!\n")    
    return classes, X, Y, Xt, Yt 

# load images directly from images folder (ex: cropped/5/)
def load_dataset_windows(train_path, height=None, width=None, test=None, shuffled=False, validation=0, 
                         mean=False, gray=False, save_dd=False, dataset_file=None, data_aug=[]):
    """ 
    Given a folder containing images separated by sub-folders (classes) returns training and testing
    data, if specified.
    """
    classes = X = Y = Xtr = Ytr = Xte = Yte = means_xtr = means_xte = None

    print("[INFO] Loading dataset (from directory)...")
    if(width and height):
        X, Y = build_image_dataset_from_dir(train_path, resize=(width, height), convert_gray=gray, 
                                            filetypes=[".bmp", ".ppm", ".jpg", ".png"], 
                                            shuffle_data=False, categorical_Y=True)
    else:
        X, Y = build_image_dataset_from_dir(train_path, resize=(width, height), convert_gray=gray, 
                                            filetypes=[".bmp", ".ppm", ".jpg", ".png"], 
                                            shuffle_data=False, categorical_Y=True)
    
    # get images dimensions automatically
    try:
        height, width, ch = X[0].shape            
    except:
        height, width = X[0].shape
        ch = 1  

    # get number of images and classes automatically
    nimages, classes = Y.shape

    print("[INFO]        Images:", nimages)

    # /////////////////////////// data augmentation process //////////////////////////////////
    if(len(data_aug) > 0):

        max_rotates = 4     # max number of rotates
        angle_step  = 90    # step for circular rotates
        max_angle   = 10    # random rotate max angle

        # list of angles to rotate
        angle_list = random.sample(range(-max_angle, max_angle), max_rotates)                        
        angle_list = np.linspace(0, 360, 360//angle_step, endpoint=False)

        flip = flop = rotate = allt = False     # enable/disable rotation data augmentation
        x_len = len(X)                          # total number of images

        # assert each 
        allt = any('all' in op.lower() for op in data_aug)
        if(not allt):
            flip   = any('flip' in op.lower() for op in data_aug)
            flop   = any('flop' in op.lower() for op in data_aug)
            rotate = any('rot' in op.lower() for op in data_aug)

        print("[INFO] Data augmentation operation...")
        print("[INFO]        Rotate:", rotate or allt)
        print("[INFO]          Flop:", flop or allt)
        print("[INFO]          Flip:", flip or allt)
        
        # apply data-augmentation to all images
        for index in range(x_len):
            if(rotate or allt):
                for angle in angle_list:
                    # create rotated image from original image
                    new_elem = scipy.ndimage.interpolation.rotate(X[index], angle, reshape=False)
                    # appends new elemnt on both X and Y
                    X.append(new_elem)
                    Y = np.concatenate([Y, [Y[index]]])  

            if(flip or allt):
                new_elem = np.flipud(X[index])
                # appends new elemnt on both X and Y
                X.append(new_elem)
                Y = np.concatenate([Y, [Y[index]]])

            if(flop or allt):
                new_elem = np.fliplr(X[index])
                # appends new elemnt on both X and Y
                X.append(new_elem)
                Y = np.concatenate([Y, [Y[index]]])

        # get number of images and classes automatically
        nimages, classes = Y.shape

        print("[INFO] Images w/ aug:", nimages)
        print("[INFO] Data augmentation operation done...")

    # /////////////////////////////////// validation split ////////////////////////////////////////////
    if(validation > 0 and validation <= 1):  # validation = [0, 1] float
        counts  = [0] * classes             # create an array to store the number of images per class

        for i in range(0, nimages):
            counts[np.argmax(Y[i])] += 1    # counts the number of images per every class
        
        print("[INFO]        Counts:", counts)
        print("[INFO]       Classes:", classes)

        # show data distribution
        if(save_dd or False):
            ax = plt.subplot(111)
            indices = range(len(counts))
            indices_str = [str(ind) for ind in indices]
            ax.bar(indices, counts, align='center')
            
            image_title = train_path.split("\\")
            image_title.reverse()
            image_title = image_title[1]
            plt.title("Images distribution per class %s" % train_path, fontweight='bold')
            plt.title("Images per class distribution on\n%s" % "German Traffic Signs dataset", fontweight='bold', fontsize=23)

            for i, v in enumerate(counts):
                ax.text((i - 0.4), (v+30), str(v), rotation=60)

            plt.xticks(indices, indices_str)
            plt.grid(True)

            ax.set_xticklabels(indices_str)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_xlabel("Class", fontsize=15)
            ax.set_ylabel("Counts", fontsize=15)

            plt.tight_layout()
            plt.savefig("%s.pdf" % image_title, dpi=300, format='pdf')
            plt.waitforbuttonpress()
            plt.show()

        Xtr = []
        Xte = []
        Ytr = []
        Yte = []

        # split train and test data manually, according to the value of the validation set
        it = 0
        for i in range(0, classes):
            it = 0 
            it += sum(counts[j] for j in range(0, i))
            
            per_class = counts[i]                           # gets the number of images per class
            to_test   = math.ceil(validation * per_class)   # calculates how many images to test per class
            split     = it + per_class - to_test            # calculates the index that splits data in train/test
            
            if False: print("%4d %4d %4d" % (it, split, split+to_test))

            Xtr += X[it:split]
            Ytr = np.concatenate([Ytr, Y[it:split]]) if len(Ytr)>0 else Y[it:split] 
            
            Xte += X[split:split+to_test]
            Yte = np.concatenate([Yte, Y[split:split+to_test]]) if len(Yte)>0 else Y[split:split+to_test]

    # /////////////////////////////////////////////////////////////////////////////////////////////////////
    else:
        Xtr = X
        Ytr = Y
        Xte = None
        Yte = None
    
    del(X)
    del(Y)

    # collect mean pixel value of each image
    if(mean):
        means_xtr = []
        means_xte = []
        
        # training images
        for img in Xtr:
            means_xtr.append(np.mean(img))
        
        means_xtr = np.array(means_xtr)
        means_xtr = np.reshape(means_xtr, (-1, 1))

        # validation images
        for img in Xte:
            means_xte.append(np.mean(img))
        
        means_xte = np.array(means_xte)
        means_xte = np.reshape(means_xte, (-1, 1))

    Xtr = np.array(Xtr)                         # convert train images list to array
    Ytr = np.array(Ytr)                         # convert train labels list to array
    if(Xte is not None): Xte = np.array(Xte)    # convert test images list to array
    if(Yte is not None): Yte = np.array(Yte)    # convert test labels list to array

    if(shuffled):
        Xtr, Ytr = shuffle(Xtr, Ytr)                                            # shuflles training data
        if(Xte is not None and Yte is not None): Xte, Yte = shuffle(Xte, Yte)   # shuflles validation data

    Xtr = np.reshape(Xtr, (-1, height, width, ch))                           # reshape array to fit on network format
    if(Xte is not None): Xte = np.reshape(Xte, (-1, height, width, ch))      # reshape array to fit on network format

    print("[INFO]          Path:", train_path)
    print("[INFO] Shape (train):", Xtr.shape, Ytr.shape)
    if(validation > 0): print("[INFO] Shape   (val):", Xte.shape, Yte.shape)
    if(mean):           print("[INFO]         Means:", means_xtr.shape, means_xte.shape)
    print("[INFO] Data loaded!\n")

    return classes, Xtr, Ytr, height, width, ch, Xte, Yte, means_xtr, means_xte

# load test images from a directory
def load_test_images(testdir=None, resize=None, mean=False, gray=False, to_array=False):
    """
    Function that loads a set of test images saved by class in distinct folders.
    Returns a list of PIL images an labels.
    """
    image_list = []
    label_list = []
    means_xte = []
    classid = 0

    channels = 3
    if(gray):
        channels = 1
    
    if(testdir is not None):
        print("[INFO] Loading test images...")
        # get all directories from testdir 
        dirs = sorted(os.walk(testdir).__next__()[1], key=numericalSort)
        
        # for each directory, get all the images inside it (same class)
        for d in dirs:
            #print(colored("\t%s" % d, "yellow"))    # NOTE: just to confirm
            tdir = os.path.join(testdir, d)
            images = os.walk(tdir).__next__()[2]
            
            # for each image, load and append it to the images list 
            for image in images:
                if image.endswith((".bmp", ".jpg", ".ppm", ".png")):
                    image_path = os.path.join(tdir, image)
                    image      = Image.open(image_path)
                    if(gray): image = image.convert('L')
                    if(resize is not None):
                        image = image.resize(resize, Image.ANTIALIAS)
                        #print("\Resized: ", image.size)
                    if(mean is not None):
                        m = np.mean(np.array(image))
                        means_xte.append(m)
                        #print("\t  Mean: ", m)

                    image_list.append(image)
                    label_list.append(classid)
            classid += 1
        
        if(to_array and resize is not None):
            lil = len(image_list)
            image_array  = np.empty((lil, resize[1], resize[0], channels), dtype=np.float32)
            labels_array = np.empty((lil, classid))

            for i in range(lil):
                image_array[i]  = np.array(image_list[i].getdata()).reshape(resize[1], resize[0], channels)
                temp = np.zeros(classid)
                temp[label_list[i]] = 1
                labels_array[i] = temp 
        
        print("\t  Path: ", testdir)
        if(to_array and resize is not None):
            print("\tImages: ", image_array.shape)
            print("\tLabels: ", labels_array.shape)
        else:
            print("\tImages: ", len(image_list))
            print("\tLabels: ", len(label_list))
        
        if(mean):
            means_xte = np.array(means_xte)
            means_xte = np.reshape(means_xte, (-1, 1))
            print("\t Mean: ", means_xte.shape)
        
        print("[INFO] Test images loaded...\n")
    else:
        sys.exit(colored("[WARNING] Path to test image is not set\n", "yellow"))
    
    # NOTE: if needed change to return lists
    if(to_array and resize is not None):
        return image_array, labels_array, means_xte
    else:
        return image_list, label_list, means_xte

# load test images from an index file
def load_test_images_from_index_file(testdir=None, infile=None):
    """
    Function that loads a set of test images according to a indexing file 
    with format: "path_to_image class_id"
    """
    image_list = []
    label_list = []
    index = 0
    
    if(testdir):
        print("[INFO] Loading test images from index file...")

        try:
            data = np.genfromtxt(infile, delimiter=", ", comments='#', names=True, 
                                skip_header=0, autostrip=True)
        except:
            sys.exit(colored("[WARNING] Index file to test images is not set", "yellow"))
        
        column = data.dtype.names[1] # 'ClassId'

        # picks every sa
        for root, dirs, files in os.walk(testdir):
            for file in files:
                if file.endswith((".bmp", ".jpg", ".ppm", ".png")):
                    image_path = os.path.join(root, file)
                    image      = Image.open(image_path).resize((32, 32))

                    #print("Image:", image_path)
                    #print("Label:", int(data[column][index]))
                    #input("")
                    
                    image_list.append(image)
                    label_list.append(int(data[column][index]))
                    index += 1
        
        # NOTE: make it general
        lil = len(image_list) # lenght of image's list
        new_image_list = np.empty((lil, 32, 32, 3), dtype=np.float32) # NOTE: set manually
        for i in range(lil):
            new_image_list[i] = np.array(image_list[i].getdata()).reshape(32, 32, 3)

        #image_list = np.array(image_list)
        #image_list = np.reshape(image_list, (-1, 32, 32, 3))

        print("\t  Path: ", testdir)
        print("\tImages: ", new_image_list.shape)
        print("\tLabels: ", len(label_list))
        print("Test images loaded...\n")
    else:
        print(colored("[WARNING] Path to test images is not set", "yellow"))
        
    return new_image_list, label_list

# load an image set from a single folder without subfolders and labels
def load_image_set_from_folder(datadir=None, resize=None, gray=False, extension="*.*"):
    images_list = [] 

    print("[INFO] Loading test images from folder...")
    try:
        filenames = sorted(glob.glob(datadir + extension), key=numericalSort)
    except:
        sys.exit(colored("[WARNING] Couldn't load test images\n", "yellow"))

    if(gray): 
        channels = 1
    else:     
        channels = 3

    for infile in filenames:
        if(gray): 
            img = Image.open(infile).convert("L")
        else:     
            img = Image.open(infile).convert("RGB")
        
        if(resize):
            img = img.resize(resize, Image.ANTIALIAS)
        images_list.append(img)
    
    # lenght of image's list
    lil = len(images_list)      
    new_images_list = np.empty((lil, resize[1], resize[0], channels), dtype=np.float32)
    for i in range(lil):
        new_images_list[i] = np.array(images_list[i].getdata()).reshape(resize[1], resize[0], channels)
    
    print("[INFO]   Path:", datadir)
    print("[INFO] Images:", new_images_list.shape)
    print("[INFO] Test images loaded...\n")
    
    return new_images_list, filenames

# function to change images colorspace
def convert_images_colorspace(images_array=None, fromc=None, convert_to=None):
    """
    Minimalist function to convert images colorspaces. From RGB 
    to other colorspace (HSV, YCrCb, YUV, ...)

    Params:
        `images_array` - images to be converted
        `fromc` - current input images colorspace
        `convert_to` - colorspace that images will be converted
    
    Return: Converted images array if convert_to is a valid
    colorspace.

    TODO: Add option from/to
    """
    new_images_array = []
    nchannels = 3

    if(convert_to is not None):
        if(convert_to == 'HSV'):
            ccode = cv2.COLOR_RGB2HSV
            nchannels = 2
        elif(convert_to == 'YCrCb'):
            ccode = cv2.COLOR_RGB2YCrCb
            nchannels = 1
        elif(convert_to == 'YUV'):
            ccode = cv2.COLOR_RGB2YUV
        elif(convert_to == 'Gray'):
            ccode = cv2.COLOR_RGB2GRAY
            nchannels = 1
        else:
            print(colored("[WARNING] Unknown colorspace %s! Returned original images." % convert_to, "yellow"))
            return images_array
        
        lia = len(images_array) # length of images array
        for i in range(lia):
            converted_image = cv2.cvtColor(images_array[i], ccode)
            new_images_array.append(converted_image[:,:,0])

        print(colored("[INFO] Converted images to colorspace %s" % convert_to, "yellow"))
    else:
        print(colored("[WARNING] No colorspace selected! Returned original images.", "yellow"))

    new_images_array = np.array(new_images_array)
    new_images_array = np.reshape(new_images_array, (-1, 32, 32, nchannels))

    print("[INFO] Converted images shape", new_images_array.shape)

    return new_images_array, nchannels

# funtion to load CIFAR-10 dataset
def load_cifar10_dataset(data_dir=None):
    from tflearn.datasets import cifar10
    from tflearn.data_utils import to_categorical
    
    HEIGHT   = 32 
    WIDTH    = 32
    CHANNELS = 3
    CLASSES  = 10

    (X, Y), (Xv, Yv) = cifar10.load_data(dirname=data_dir, one_hot=True)
    X, Y = shuffle(X, Y)
    Xv, Yv = shuffle(Xv, Yv)

    Xt = Xv[2000:]
    Yt = Yv[2000:]

    Xv = Xv[:2000]
    Yv = Yv[:2000]

    return CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, Xt, Yt

# funtion to load MNIST dataset
def load_mnist_dataset(data_dir=None):
    import tflearn.datasets.mnist as mnist

    HEIGHT   = 28 
    WIDTH    = 28
    CHANNELS = 1
    CLASSES  = 10

    X, Y, Xv, Yv = mnist.load_data(data_dir=data_dir, one_hot=True)
    X, Y = shuffle(X, Y)
    Xv, Yv = shuffle(Xv, Yv)
    X = X.reshape([-1, 28, 28, 1])
    Xv = Xv.reshape([-1, 28, 28, 1])

    Xt = Xv[2000:]
    Yt = Yv[2000:]

    Xv = Xv[:2000]
    Yv = Yv[:2000]

    return CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, Xt, Yt