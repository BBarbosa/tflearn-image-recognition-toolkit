from __future__ import division, print_function, absolute_import

import tflearn
import winsound as ws  # for windows only
import tensorflow as tf
import scipy.ndimage
import os,sys,time,platform,six

from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data
from tflearn.data_augmentation import ImageAugmentation
from utils import architectures

import scipy.ndimage
import numpy as np 
from PIL import Image,ImageDraw

from matplotlib import pyplot as plt


HEIGHT = WIDTH = 128

def load_model(arch,classes,modelpath):
    # model definition
    network = input_data(shape=[None, HEIGHT, WIDTH, 3],     # shape=[None,IMAGE, IMAGE] for RNN
                    data_preprocessing=None,       
                    data_augmentation=None)

    network = architectures.build_network(arch,network,classes)
    
    model = tflearn.DNN(network) # tensorboard_dir='logs'

    print("Loading trained model...")  
    model.load(modelpath)
    print("\tModel: ",modelpath)
    print("Trained model loaded!\n")
    
    return model

# computational resources definition
tflearn.init_graph(num_cores=8,gpu_memory_fraction=0.9)

# fix for working on windows
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

# graph for model 1
graph1 = tf.Graph()
with graph1.as_default():
    #with tf.Session(graph=graph1) as sess1:
    model1 = load_model("mycifar",7,"models\\aa7c.tflearn")
    print("*"*10,"Model 1 loaded!","*"*10)

# graph for model 2
graph2 = tf.Graph()
with graph2.as_default():
    #with tf.Session(graph=graph2) as sess2:
    model2 = load_model("mycifar",3,"models\\aa678.tflearn") 
    print("*"*10,"Model 2 loaded!","*"*10)

input("All good. Press to continue...")                