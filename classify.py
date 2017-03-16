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

# function to display convolutions
def display_convolutions(model, layer, padding=4, filename='', nrows=1):
    if isinstance(layer, six.string_types):
        vars = tflearn.get_layer_variables_by_name(layer)
        variable = vars[0]
    else:
        variable = layer.W

    data = model.get_weights(variable)

    # N is the total number of convolutions
    N = data.shape[2] * data.shape[3]

    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[0]
    # Size of the result image including padding
    result_size = filters_per_row * (filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[3]):
        for c in range(data.shape[2]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    result[filter_y * (filter_size + padding) + i, filter_x * (filter_size + padding) + j] = \
                        data[i, j, c, n]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)

    
    # Plot figure according to the number of rows to show
    limit = filter_size * nrows + (nrows-1) * padding   # limit the number of lines to show on figure

    plt.figure(figsize=(16,16))
    plt.axis('off')
    plt.imshow(result[0:limit], cmap='gray', interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    plt.show()

# script arguments' check
if(len(sys.argv) < 4):
    print(colored("Call: $ python classify.py {architecture} {model} {image} {classid}","yellow"))
    sys.exit(colored("ERROR: Not enough arguments!","yellow"))
else:
    # specify OS
    OS = platform.system() 
    
    # clear screen and show OS
    if(OS == 'Windows'):
        os.system('cls')
    else:
        os.system('clear')
    print("Operating System --> %s\n" % OS)

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
    network = input_data(shape=[None, WIDTH, HEIGHT, 3],     # shape=[None,IMAGE, IMAGE] for RNN
                        data_preprocessing=None,       
                        data_augmentation=None) 

    network = architectures.build_network(arch,network,classes)

    # fix for Windows
    if(OS == 'Windows'):
        col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for x in col:
            tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

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
                
