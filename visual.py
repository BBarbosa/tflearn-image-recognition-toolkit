from __future__ import division, print_function, absolute_import

import sys,os,platform,six
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflearn
import tensorflow as tf
import numpy as np
import math,scipy
import matplotlib.pyplot as plt
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data
from utils import architectures
from colorama  import init
from termcolor import colored
from scipy import ndimage,misc,interpolate
from PIL import Image
from multiprocessing import Process

# init colored print
init()

# plot_weights(weights, input_channel=0) working good
def plot_conv_weights(weights, input_channel=0):
    # argument: weigths = conv1.W
    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]
    print("  num_fitlers: ", num_filters)

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    # 32 filters » minimum grid 6x6 (=36) 
    num_grids = math.ceil(math.sqrt(num_filters))
    print("         grid: ", num_grids, "x", num_grids,"\n")

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
            if(input_channel == 3):
                kernel = weights[:, :, :, i]
            else:
                kernel = weights[:, :, input_channel, i]

            kernel = scipy.misc.imresize(kernel,(32,32),interp='cubic')

            # Plot image.
            ax.imshow(kernel, vmin=None, vmax=None,interpolation='nearest',cmap='gray')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    fig.show()

# plot_conv_layer(layer, image) NOTE: ERROR. Must be fixed
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
            img = values[:, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# shows the result of applying N learnt filters to an image
def convolve_filters(image,weights,max_filters=None,input_channel=0):
    """
    Function that convolves N filters (a set of weights correspond to
    a bunch of filters) in one image
    """
    # argument: weights = conv1.W
    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]
    print("  num_fitlers: ", num_filters)

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    # 32 filters » minimum grid 6x6 (=36) 
    num_grids = math.ceil(math.sqrt(num_filters))
    print("         grid: ", num_grids, "x", num_grids,"\n")

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
            if(input_channel == 3):
                kernel = weights[:, :, :, i]
            else:
                kernel = weights[:, :, input_channel, i]

            img = ndimage.convolve(image,kernel,mode='constant')

            # Plot image.
            ax.imshow(img, vmin=None, vmax=None,interpolation='nearest',cmap='gray')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# NOTE: Not working properly
# nice image printer 
# Note: matplot lib is pretty inconsistent with how it plots these weird image arrays.
# Try running them a couple of times if the output doesn't quite match the blog post results.
def nice_image_printer(model, image):
    """
    Prints the image as a 2d array
    """
    image_batch = np.expand_dims(image,axis=0)
    image_batch = np.reshape(image_batch,(1,HEIGHT,WIDTH,3))
    conv_image2 = model.predict(image_batch)

    conv_image2 = np.squeeze(conv_image2, axis=0)
    print(conv_image2.shape)
    conv_image2 = conv_image2.reshape(conv_image2.shape[:2])

    print(conv_image2.shape)
    plt.imshow(conv_image2)
    
"""
Script definition
"""

if (len(sys.argv) < 5):
    print(colored("Call: $ python visual.py {architecture} {model} {layer} {input_channel} [image]","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))

# specify OS
OS = platform.system() 

# clear screen and show OS
if(OS == 'Windows'):
    os.system('cls')
else:
    os.system('clear')
print("Operating System: %s\n" % OS)

# NOTE: images properties 
HEIGHT   = 64
WIDTH    = 64
CHANNELS = 1
CLASSES  = 11

# get command line arguments
arch      = sys.argv[1]       # name of architecture
modelpath = sys.argv[2]       # path to saved model
layer     = sys.argv[3]       # layer name, for example, Conv2D or Conv2D_1
ichannel  = int(sys.argv[4])  # input channel for displaying kernels and convolutions

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_samplewise_zero_center()   # per sample (featurewise is a global value)
img_prep.add_samplewise_stdnorm()       # per sample (featurewise is a global value)

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=10.)

# network definition
network = input_data(shape=[None, WIDTH, HEIGHT, CHANNELS],     # shape=[None,IMAGE, IMAGE] for RNN
                    data_preprocessing=None,       
                    data_augmentation=None) 

#in2 = input_data(shape=[None,1])
#print(network.shape)
#print(in2.shape,"\n")
#network = architectures.build_merge_test(network,in2,CLASSES)

network,_ = architectures.build_network(arch,network,CLASSES)

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
plot_conv_weights(weights,input_channel=ichannel)

# weights[:, :, input_channel, i]
# print(" Filter shape: ", weights[:,:,:,0].shape)

# choose kernel
# kernel = weights[:,:,:,2]

# tries to load a image
load = True
try:
    if(ichannel == 3):
        img = scipy.ndimage.imread(sys.argv[5],mode='RGB', flatten=False)
    else:
        img = scipy.ndimage.imread(sys.argv[5],mode='L', flatten=True)
    
    plt.imshow(img,cmap='gray')
except:
    load = False
    sys.exit(colored("WARNING: Image not mentioned!","yellow"))

# if loaded image correctly
if(load):
    img  = scipy.misc.imresize(img, (HEIGHT,WIDTH), interp="bicubic").astype(np.float32)
    #img -= scipy.ndimage.measurements.mean(img)       # confirmed. check data_utils.py on github
    #img /= np.std(img)                                # confirmed. check data_utils.py on github
    #img = np.array(img)
    #img = np.reshape(img,(-1,HEIGHT,WIDTH,3))

    """
    Pre-process image?
    """
    print("  Image shape: ",img.shape, type(img))
    convolve_filters(img,weights,input_channel=ichannel)
    #nice_image_printer(model,img)