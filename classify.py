from __future__ import division, print_function, absolute_import

import os,sys,time,platform,six
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflearn
import winsound as ws  # for windows only
import tensorflow as tf
import scipy.ndimage

from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data
from tflearn.data_augmentation import ImageAugmentation
from utils import architectures

import scipy.ndimage
import numpy as np 
from PIL import Image,ImageDraw

from matplotlib import pyplot as plt
from colorama import init
from termcolor import colored

# init colored print
init()

# return a color according to the class
def getColor(x):
    return {
        0 : (255,0,0),      # red
        1 : (0,255,0),      # green
        2 : (0,0,255),      # blue
        3 : (255,255,0),    # yellow
        4 : (255,128,5),    # orange  
        5 : (255,20,147),   # pink
        6 : (0,255,255),    # cyan
        7 : (255,255,255),  # white
        8 : (128,0,128),    # purple
    }[x]


# script arguments' check
if(len(sys.argv) < 4):
    print(colored("Call: $ python classify.py {architecture} {model} {image} {classid}","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))
else:
    # specify OS
    OS = platform.system() 
    
    # clear screen and show OS
    if(OS == 'Windows'):
        os.system('cls')
    else:
        os.system('clear')
    print("Operating System: %s\n" % OS)

    # images properties (inherit from trainning?)   
    HEIGHT  = 306
    WIDTH   = 259
    classes = 17

    # get command line arguments
    arch      = sys.argv[1]       # name of architecture
    modelpath = sys.argv[2]       # path to saved model
    filename  = sys.argv[3]       # test image name/path
    classid   = int(sys.argv[4])  # test image class id. -1 for collages

    # a bunch of flags
    showConvolution = False

    # computational resources definition
    tflearn.init_graph(num_cores=8,gpu_memory_fraction=0.9)

    # network definition
    network = input_data(shape=[None, HEIGHT, WIDTH, 3],     # shape=[None,IMAGE, IMAGE] for RNN
                        data_preprocessing=None,       
                        data_augmentation=None) 

    network = architectures.build_network(arch,network,classes)

    # model definition
    model = tflearn.DNN(network, checkpoint_path='models',
                        max_checkpoints=1, tensorboard_verbose=0) # tensorboard_dir='logs'


    print("Loading trained model...")  
    model.load(modelpath)
    print("\tModel: ",modelpath)
    print("Trained model loaded!\n")

    # display convolutions in a figure
    if(showConvolution):
        layer = "Conv2D"
        display_convolutions(model,layer,padding=4,filename='',nrows=4)

    # Load the test image 
    img   = scipy.ndimage.imread(filename, mode='RGB')     # mode='L', flatten=True -> grayscale
    img   = scipy.misc.imresize(img, (wDIM,hDIM), interp="bicubic").astype(np.float32, casting='unsafe')
    img  -= scipy.ndimage.measurements.mean(img)           # confirmed. check data_utils.py on github
    img  /= np.std(img)                                    # confirmed. check data_utils.py on github
        
    #img = featurewise_zero_center(img, mean=None)      # from data_utils.py
    #img = featurewise_std_normalization(img, std=None)  # from data_utils.py

    print("Classification started...")
    print("\tImage: ", filename)
    print("\t Size: ", wDIM, "x", hDIM)
    
    # start measuring time
    start_time = time.time()
    img = np.reshape(img,(1,HEIGHT,WIDTH,3))
    
    probs = model.predict(img)
    index = np.argmax(probs)

    ctime = time.time() - start_time
    
    print("\t Time:  %s seconds" % ctime)
    print("Classification done!\n")

    # assuming modelpath: "models\name.tflearn" -> name
    try:
        modelname = modelpath.split("\\")[1].split(".")[0]
    except:
        modelname = modelpath.split("/")[1].split(".")[0]
    
    # Error check if not a collage
    if (classid != -1):
        print("Error check...")
        print("\t  Image: ", filename)
        print("\t  Class: ", classid)
        print("\tPredict: ", index)
        print("Error checked!\n")

        error_file = "%s_%s_error.txt" % (modelname,arch)
        ferror = open(error_file, "a+")
                 
        ferror.write("Class: %d | Predicted: %d | File: %s | Time: %f\n" % (classid,index,filename,ctime))
        ferror.close()

    if(OS == 'Windows'):
        freq = 2000
        dur  = 1000 
        ws.Beep(freq,dur)
                
