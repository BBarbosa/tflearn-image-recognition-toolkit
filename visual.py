from __future__ import division, print_function, absolute_import

import sys,os,platform,six
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflearn
import tensorflow as tf
import numpy as np
import math,scipy
import matplotlib.pyplot as plt

from tflearn.layers.core import input_data
from utils import architectures
from colorama  import init
from termcolor import colored

# init colored print
init()

# plot_weights(weights, input_channel=0)
def plot_conv_weights(weights, input_channel=0):
    # argument: weigths = conv1.W
    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]
    print("  num_fitlers: ", num_filters)

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    # 32 filters » minimum grid 6x6 (=36) 
    num_grids = math.ceil(math.sqrt(num_filters))
    print("         grid: ", num_grids, "x", num_grids)

    # Create figure with a grid of sub-plots. 
    # grid 6x6
    fig, axes = plt.subplots(num_grids, num_grids)
    axes = np.array(axes)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=None, vmax=None,
                      interpolation='nearest', cmap='gray')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# plot_conv_layer(layer, image) ERROR
def plot_conv_layer(layer, image):
    # argument: layer_conv1 or layer_conv2.
    session = tf.Session()
    x = tf.placeholder(dtype="float", shape=[None,HEIGHT,WIDTH,3])
    k = tf.placeholder(dtype="float")

    session.run(tf.global_variables_initializer())
    print(" Tensor shape: ", x.get_shape(), type(x))
    print("  Layer shape: ", type(layer))
    print("        Layer: ", layer)
    
    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: image, k: 1.0}
    
    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)
    # Number of filters used in the conv. layer.
    num_filters = values[0].shape

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    # Same process as the filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)
    axes = np.array(axes)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# script definition
if (len(sys.argv) < 4):
    print(colored("Call: $ python visual.py {architecture} {model} {layer} [image]","yellow"))
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
    HEIGHT  = 128
    WIDTH   = 128
    CLASSES = 7

    # get command line arguments
    arch      = sys.argv[1]       # name of architecture
    modelpath = sys.argv[2]       # path to saved model
    layer     = sys.argv[3]       # layer name, for example, Conv2D or Conv2D_1

    # network definition
    network = input_data(shape=[None, WIDTH, HEIGHT, 3],     # shape=[None,IMAGE, IMAGE] for RNN
                        data_preprocessing=None,       
                        data_augmentation=None) 

    network = architectures.build_network(arch,network,CLASSES)

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
    
    # get layer by its name
    if isinstance(layer, six.string_types):
        vars = tflearn.get_layer_variables_by_name(layer)
        variable = vars[0]
    else:
        variable = layer.W
    
    # load weights (learnt filters) and plots them
    weights = model.get_weights(variable)
    print("Weights shape: ", weights.shape)
    plot_conv_weights(weights)

    # tries to load a image
    load = True
    try:
        img = scipy.ndimage.imread(sys.argv[4])
    except:
        load = False
        sys.exit(colored("ERROR: Image not mentioned!","yellow"))
    
    # if loaded image correctly
    if(load):
        img = scipy.misc.imresize(img, (HEIGHT,WIDTH), interp="bicubic").astype(np.float32)
        img = np.array(img)
        img = np.reshape(img,(-1,HEIGHT,WIDTH,3))
        print("  Image shape: ",img.shape, type(img))

        plot_conv_layer(vars,img)
        

