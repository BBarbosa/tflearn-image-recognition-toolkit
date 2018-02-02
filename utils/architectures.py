"""
Architecture definition module
"""

from __future__ import division, print_function, absolute_import

import sys
import tflearn
import numpy as np
import tensorflow as tf

from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, highway_conv_2d, avg_pool_2d, upsample_2d, upscore_layer, conv_2d_transpose
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from colorama import init
from termcolor import colored

# init colored print
init()

# vgg16 (heavy) 
def build_vgg16(network, classes):
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='rmsprop', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001)
    return network

def build_vgg16_ft(network, classes):
    x = tflearn.conv_2d(network, 64, 3, activation='relu', name='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', name='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', name='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', name='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', name='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', name='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, classes, activation='softmax', name='fc8', 
                                restore=False)

    regression = tflearn.regression(x, optimizer='adam', 
                                loss='categorical_crossentropy', 
                                learning_rate=0.001, restore=False)
    return network

# vgg16 (light) 
def build_myvgg16(network, classes):
    network = conv_2d(network, 32, 3, activation='relu')
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = batch_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = batch_normalization(network)

    network = conv_2d(network, 96, 3, activation='relu')
    network = conv_2d(network, 96, 3, activation='relu')
    network = conv_2d(network, 96, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = batch_normalization(network)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001)
    return network

# tyres
def build_tyres(network, classes):
    network = conv_2d(network, 8, 5, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 8, 1, activation='relu')
    network = conv_2d(network, 16, 3, activation='relu')
    network = conv_2d(network, 16, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 16, 1, activation='relu')
    network = conv_2d(network, 32, 3, activation='relu')
    network = conv_2d(network, 32, 3, activation='relu')
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    
    network = conv_2d(network, 16, 1, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = fully_connected(network, 256, activation='relu')
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.00001)
    return network

# mnist 
def build_mnist(network, classes):
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', learning_rate=0.01, 
                        loss='categorical_crossentropy', name='target')
    return network

# cifar10 
def build_cifar10(network, classes):
    network = conv_2d(network, 32, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu') 
    network = conv_2d(network, 64, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu') # 512
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.001)
    return network

# cifar10 modified
def build_cifar10_mod(network, classes):
    network = conv_2d(network, 32, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu') 
    network = conv_2d(network, 64, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.001) # 0.001
    return network

# cifar10 modified
# from fundamentals of DL book
def build_cifar10_mod2(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', regularizer="L2") 
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2") 
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2") 
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)
    
    network = flatten(network)

    network = fully_connected(network, 384, activation='relu') # 512
    #network = dropout(network, 0.5)
    network = fully_connected(network, 192, activation='relu') # 512
    #network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.01) # 0.001
    return network

# cifar10 valid
def build_cifar10_valid(network, classes):
    network = conv_2d(network, 32, 3, activation='relu', padding='valid') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu', padding='valid') 
    network = conv_2d(network, 64, 3, activation='relu', padding='valid') 
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.001)
    return network

# cifar10 x2
def build_cifar10_x2(network, classes):
    network = conv_2d(network, 64, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 128, 3, activation='relu') 
    network = conv_2d(network, 128, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 1024, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.001)
    return network

# cifar10 x0.5
def build_cifar10_half(network, classes):
    network = conv_2d(network, 8, 3, activation='relu') #16
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 16, 3, activation='relu') #32
    network = conv_2d(network, 16, 3, activation='relu') #32
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 128, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.001)
    return network

# mycifar (fixed)
def build_mycifar_my(network, classes):
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 2) 
    network = local_response_normalization(network)
    
    network = conv_2d(network, 96, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.0001
    return network

def build_mycifar(network, classes):
    network = conv_2d(network, 32, 11, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 7, activation='relu')
    network = max_pool_2d(network, 2) 
    network = local_response_normalization(network)
    
    network = conv_2d(network, 96, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.0001
    return network

# mycifarv2
def build_mycifar_v2(network, classes):
    network = conv_2d(network, 32, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 3, activation='relu')
    network = avg_pool_2d(network, 2) 
    network = local_response_normalization(network)
    
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    #sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=125) # 0.005

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00001
    return network

# mycifarv3
def build_mycifar_v3(network, classes):
    network = conv_2d(network, 32, 3, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2) 
    network = local_response_normalization(network)
    
    network = conv_2d(network, 96, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) # 0.5 -> 0.75
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.0001
    return network

# mycifarv4
def build_mycifar_v4(network, classes):
    network = conv_2d(network, 64, 7, activation='relu', strides=4) 
    #network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 7, activation='relu', strides=4) 
    #network = conv_2d(network, 96, 5, activation='relu') 
    #network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')

    #sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.97, decay_step=41) # 0.005

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mycifar_v5
def build_mycifar_v5(network, classes):
    network = conv_2d(network, 32, 11, activation='relu', strides=4) 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 7, activation='relu')
    network = max_pool_2d(network, 2) 
    network = local_response_normalization(network)
    network = conv_2d(network, 96, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.00001)    # 0.0001
    return network

# mycifar_v6
def build_mycifar_v6(network, classes):
    network = conv_2d(network, 32, 11, activation='relu', strides=4) 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 7, activation='relu')
    network = max_pool_2d(network, 2) 
    network = local_response_normalization(network)
    network = conv_2d(network, 96, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 1024, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.00001)    # 0.0001
    return network

# 1 conv layer networks --------------------------------------------------------------------
def build_1l_32f_5x5_fc512(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_1l_32f_5x5_fc512_s2(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)   
    return network

def build_1l_32f_5x5_fc412(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 412, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_32f_5x5_fc312(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 312, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc212(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc112(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 112, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc50(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc25(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 25, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc10(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 10, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc5(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 5, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_32f_5x5_fc4(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 4, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_32f_5x5_fc2(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 2, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_32f_5x5_fc1(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 1, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_24f_5x5_fc50(network, classes):
    network = conv_2d(network, 24, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_16f_5x5_fc50(network, classes):
    network = conv_2d(network, 16, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_8f_5x5_fc50(network, classes):
    network = conv_2d(network, 8, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam',  
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_1l_4f_5x5_fc50(network, classes):
    network = conv_2d(network, 4, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_8f_3x3_fc50(network, classes):
    network = conv_2d(network, 8, 3, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_8f_7x7_fc50(network, classes):
    network = conv_2d(network, 8, 7, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_8f_9x9_fc50(network, classes):
    network = conv_2d(network, 8, 9, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network
 
def build_1l_8f_11x11_fc50(network, classes):
    network = conv_2d(network, 8, 11, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_4f_5x5_fc50_nd(network, classes):
    network = conv_2d(network, 4, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    # same as v114 but without dropout
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_2f_5x5_fc50(network, classes):
    network = conv_2d(network, 2, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_1f_5x5_fc50(network, classes):
    network = conv_2d(network, 1, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# //////////////////// 2 conv layers network ////////////////////
def build_2l_8f_16f_5x5_fc512(network, classes):
    network = conv_2d(network, 8, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 16, 5, activation='relu', strides=2)  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.001    
    return network

def build_2l_8f_16f_5x5_fc256(network, classes):
    network = conv_2d(network, 8, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = fully_connected(network, 256, activation='relu') 
    network = dropout(network, 0.75)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.001    
    return network

def build_2l_8f_16f_3x3_fc256(network, classes):
    network = conv_2d(network, 8, 3, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 3, activation='relu')  
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = fully_connected(network, 256, activation='relu') 
    network = dropout(network, 0.75)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001) # 0.001    
    return network

def build_2l_8f_16f_5x5_fc256_ns(network, classes):
    network = conv_2d(network, 8, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = fully_connected(network, 256, activation='relu') 
    network = dropout(network, 0.75)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.001    
    return network

def build_2l_32f_5x5_fc512(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu', strides=2)  
    #network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.001    
    return network


def build_2l_32f_5x5_fc512_ns(network, classes):
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_2l_32f_64f_3x3_fc512(network, classes):
    network = conv_2d(network, 32, 3, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu', strides=2)  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_2l_32f_64f_3x3_fc512_ns(network, classes):
    network = conv_2d(network, 32, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_2l_32f_64f_5x5_fc512(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2)
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu', strides=2)
    #network = conv_2d(network, 8, 1, activation='relu')  
    #network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.0001    
    return network

def build_2l_32f_64f_5x5_fc512_min(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2)
    network = conv_2d(network, 8, 1, activation='relu')
    network = max_pool_2d(network, 2)
    
    network = conv_2d(network, 64, 5, activation='relu')
    network = conv_2d(network, 16, 1, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.0001    
    return network

def build_2l_32f_64f_5x5_fc512_ns(network, classes):
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

# 3 conv layer network -------------------------------------------------------------------
def build_3l_32f_3x3_fc512(network, classes):
    network = conv_2d(network, 32, 3, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 32, 3, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network


def build_3l_32f_3x3_fc512_ns(network, classes):
    network = conv_2d(network, 32, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 3, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_3l_32f_5x5_fc512(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 32, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network


def build_3l_32f_5x5_fc512_ns(network, classes):
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 32, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_3l_8f_16f_16f_5x5_fc256(network, classes):
    network = conv_2d(network, 8, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 16, 5, activation='relu') 
    network = conv_2d(network, 16, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 256, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.0001    
    return network

def build_3l_16f_32f_32f_5x5_fc512(network, classes):
    network = conv_2d(network, 16, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu') 
    network = conv_2d(network, 32, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.0001    
    return network


def build_3l_32f_64f_64f_3x3_fc512(network, classes):
    network = conv_2d(network, 32, 3, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)  
    
    network = conv_2d(network, 64, 3, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.0001   
    return network


def build_3l_32f_64f_64f_3x3_fc512_ns(network, classes):
    network = conv_2d(network, 32, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 64, 3, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')


    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001)    
    return network

def build_3l_32f_64f_64f_5x5_fc512(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 64, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.0001    
    return network

def build_3l_32f_64f_64f_533_fc512(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 64, 3, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) # 0.0001    
    return network

def build_3l_32f_64f_64f_5x5_fc256_fc512(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 64, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 256, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_3l_32f_64f_64f_5x5_fc512_ns(network, classes):
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)  
    
    network = conv_2d(network, 64, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001)    
    return network

def build_3l_32f_64f_128f_5x5_fc512(network, classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 128, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network


# 4 conv layer network -------------------------------------------------------------------
def build_4l_32f_5x5_fc512(network, classes):
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 32, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_4l_32f_5x5_fc512_ns(network, classes):
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu')  
    network = conv_2d(network, 32, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_4l_16f_32f_48f_64f_5x5_fc512(network, classes):
    network = conv_2d(network, 16, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 48, 5, activation='relu')
    network = max_pool_2d(network, 2)  
    network = conv_2d(network, 64, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_4l_32f_32f_64f_64f_5x5_fc512_ns(network, classes):
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu')  
    network = conv_2d(network, 64, 5, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network
# others networks -------------------------------------------------------------------------
# all cnn
def build_all_cnn(network, classes):
    #network = conv_2d(network, 96, 3, activation='relu')
    #network = conv_2d(network, 96, 3, activation='relu')
    network = conv_2d(network, 96, 3, activation='relu', strides=2) 
    #network = dropout(network, 0.5)
    #network = conv_2d(network, 192, 3, activation='relu')
    #network = conv_2d(network, 192, 3, activation='relu')
    network = conv_2d(network, 192, 3, activation='relu', strides=2)
    #network = dropout(network, 0.5)
    network = conv_2d(network, 192, 3, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu') 

    network = avg_pool_2d(network, 4)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')

    #sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.97, decay_step=41) # 0.005

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.00005)    # 0.00005
    return network

# alexnet
def build_alex(network, nclasses):
    network = conv_2d(network, nb_filters=96, filter_size=11, strides=4, activation='relu', padding='same')
    network = max_pool_2d(network, kernel_size=3, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, nb_filters=256, filter_size=5, strides=1, activation='relu', padding='same')
    network = max_pool_2d(network, kernel_size=3, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, nb_filters=384, filter_size=3, strides=1, activation='relu', padding='same')
    network = conv_2d(network, nb_filters=384, filter_size=3, strides=1, activation='relu', padding='same')
    network = conv_2d(network, nb_filters=256, filter_size=3, strides=1, activation='relu', padding='same')
    network = max_pool_2d(network, kernel_size=3, strides=2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, n_units=4096, activation='relu')
    network = dropout(network, keep_prob=0.5)
    network = fully_connected(network, n_units=4096, activation='relu')
    network = dropout(network, keep_prob=0.5)
    network = fully_connected(network, n_units=nclasses, activation='softmax')
    
    network = regression(network, optimizer='momentum', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001)
    return network 

# my alexnet
def build_myalex(network, classes):
    network = conv_2d(network, 96, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 1024, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 1024, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='momentum', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)
    return network 


# network in network (error)
def build_nin(network, classes):
    network = conv_2d(network, 192, 5, activation='relu')
    network = conv_2d(network, 160, 1, activation='relu')
    network = conv_2d(network, 96, 1, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = dropout(network, 0.5)
    
    network = conv_2d(network, 192, 5, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = avg_pool_2d(network, 3, strides=2)
    network = dropout(network, 0.5)
    
    network = conv_2d(network, 192, 3, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = conv_2d(network, 10, 1, activation='relu')
    network = avg_pool_2d(network, 8)
    
    network = flatten(network)
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='softmax_categorical_crossentropy', 
                        learning_rate=0.001)
    return network

# cnn highway (error)
def build_highway(network, classes):
    for i in range(3):
        for j in [3, 2, 1]: 
            network = highway_conv_2d(network, 16, j, activation='elu')
        network = max_pool_2d(network, 2)
        network = batch_normalization(network)
        
    network = fully_connected(network, 128, activation='elu')
    network = fully_connected(network, 256, activation='elu')
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', learning_rate=0.01, 
                        loss='categorical_crossentropy', name='target')
    return network

# rnn
def build_rnn(network, classes):

    network = tflearn.lstm(network, 128, return_seq=True)
    network = tflearn.lstm(network, 128)
    network = tflearn.fully_connected(network, classes, activation='softmax')
    
    network = tflearn.regression(network, optimizer='adam', 
                            loss='categorical_crossentropy', name="output1")
    return network

# resnet (error)
def build_resnet(network, classes, param):
    try:
        n = int(param)
    except:
        n = 5

    network = tflearn.conv_2d(network, 16, 3, regularizer='L2', weight_decay=0.0001)
    network = tflearn.residual_block(network, n, 16)
    network = tflearn.residual_block(network, 1, 32, downsample=True)
    
    network = tflearn.residual_block(network, n-1, 32)
    network = tflearn.residual_block(network, 1, 64, downsample=True)
    
    network = tflearn.residual_block(network, n-1, 64)
    network = tflearn.batch_normalization(network)
    network = tflearn.activation(network, 'relu')
    network = tflearn.global_avg_pool(network)
    
    # Regression
    network = tflearn.fully_connected(network, classes, activation='softmax')
    mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    network = tflearn.regression(network, optimizer=mom, 
                            loss='categorical_crossentropy')
    
    return network

def build_resnet_v2(network, classes, depth=5):
    n = int(depth)
    network = tflearn.conv_2d(network, 16, 3, regularizer='L2', weight_decay=0.0001)
    network = tflearn.residual_block(network, n, 16, batch_norm=False)
    network = tflearn.residual_block(network, 1, 32, downsample=True, batch_norm=False)
    
    network = tflearn.residual_block(network, n-1, 32, batch_norm=False)
    network = tflearn.residual_block(network, 1, 64, downsample=True, batch_norm=False)
    
    network = tflearn.residual_block(network, n-1, 64, batch_norm=False)
    network = tflearn.activation(network, 'relu')
    network = tflearn.global_avg_pool(network)
    
    # Regression
    network = tflearn.fully_connected(network, classes, activation='softmax')
    mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    network = tflearn.regression(network, optimizer=mom, 
                                 loss='categorical_crossentropy')
    
    return network

# error
def build_densenet(network, classes):
    # Growth Rate (12, 16, 32, ...)
    k = 12

    # Depth (40, 100, ...)
    L = 40
    nb_layers = int((L - 4) / 3)

    network = tflearn.layers.conv_2d(network, 16, 3, regularizer='L2', weight_decay=0.0001)
    network = tflearn.layers.densenet_block(network, nb_layers, k)
    network = tflearn.layers.densenet_block(network, nb_layers, k)
    network = tflearn.layers.densenet_block(network, nb_layers, k)
    network = tflearn.global_avg_pool(network)

    # Regression
    network = tflearn.fully_connected(network, classes, activation='softmax')
    opt = tflearn.optimizers.Nesterov(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    network = tflearn.regression(network, optimizer=opt, 
                                 loss='categorical_crossentropy')
    
    return network

# dlib (used on mnist)
def build_dlib(network, classes):
    network = conv_2d(network, 6, 5, activation='relu', padding='same', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 16, 5, activation='relu', padding='same')
    network = max_pool_2d(network, 2) 

    network = fully_connected(network, 120, activation='relu') 
    network = fully_connected(network, 84, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001)    
    return network

# personal small net
def build_small_net(network, classes):
    network = conv_2d(network, 8, 5, activation='relu', padding='same', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 16, 5, activation='relu', padding='same', strides=2)
    network = max_pool_2d(network, 2) 

    network = fully_connected(network, 120, activation='relu') 
    network = fully_connected(network, 84, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001, name='Regression')    
    return network

# small net
def build_snet(network, classes):
    network = conv_2d(network, 8, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 4)

    network = fully_connected(network, 128, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# merge test
def build_merge_test(network, in2, classes):
    network = conv_2d(network, 10, 2, activation='relu', strides=4)
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 128, activation='relu')
    network = merge([network, in2], mode='concat')
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# costum net
def build_custom(network, classes):
    network = conv_2d(network, 10, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 20, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 30, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    #network = flatten(network)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=0.001, 
                        loss='categorical_crossentropy', name='target')
    return network

# costum net
def build_custom2(network, classes):
    network = conv_2d(network, 10, 5, activation='relu')
    network = max_pool_2d(network, 2)
    #network = local_response_normalization(network)
    
    network = conv_2d(network, 20, 5, activation='relu')
    network = max_pool_2d(network, 2)
    #network = local_response_normalization(network)
    
    network = conv_2d(network, 30, 5, activation='relu')
    network = max_pool_2d(network, 2)
    #network = local_response_normalization(network)
    
    #network = flatten(network)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=0.001, 
                        loss='categorical_crossentropy', name='target')
    return network

# costum net
def build_custom3(network, classes):
    network = conv_2d(network, 16, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    #network = flatten(network)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=0.001, 
                        loss='categorical_crossentropy', name='target')
    return network

def build_custom31(network, classes):
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 48, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    #network = flatten(network)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=0.001, 
                        loss='categorical_crossentropy', name='target')
    return network

def build_custom32(network, classes):
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)
    
    network = conv_2d(network, 48, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)
    
    #network = conv_2d(network, 64, 3, activation='relu')
    #network = max_pool_2d(network, 2)
    #network = batch_normalization(network)
    
    #network = flatten(network)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    #network = fully_connected(network, 256, activation='relu')
    #network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=0.001, 
                        loss='categorical_crossentropy', name='target')
    return network

def build_custom4(network, classes):
    network = conv_2d(network, 16, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)
    
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)
    
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)
    
    network = flatten(network)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=0.001, 
                        loss='categorical_crossentropy', name='target')
    
    return network

# lenet net
def build_lenet(network, classes):
    network = conv_2d(network, 20, 5, activation='relu')
    network = max_pool_2d(network, 2)
    
    network = conv_2d(network, 50, 5, activation='relu')
    network = max_pool_2d(network, 2)
    
    network = flatten(network)
    
    network = fully_connected(network, 500, activation='relu')
    network = fully_connected(network, classes, activation='softmax')
    
    sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=125)

    network = regression(network, optimizer=sgd, learning_rate=0.001, 
                        loss='categorical_crossentropy', name='target')
    return network


# non-convolutional 
def build_non_convolutional(network, classes):
    network = max_pool_2d(network, 20)
    network = conv_2d(network, 10, 3, activation='relu')

    network = fully_connected(network, 256, activation='relu')
    network = fully_connected(network, classes, activation='softmax') 

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)   

    return network

# autoencoder example
def build_autoencoder(network, classes): 
    # encoder 
    encoder = conv_2d(network, 16, 7, activation='relu') 
    encoder = conv_2d(encoder, 16, 7, activation='relu')
    encoder = max_pool_2d(encoder, 2)

    encoder = conv_2d(encoder, 32, 5, activation='relu') 
    encoder = conv_2d(encoder, 32, 5, activation='relu')
    encoder = max_pool_2d(encoder, 2)

    encoder = conv_2d(encoder, 64, 3, activation='relu') 
    encoder = conv_2d(encoder, 64, 3, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    
    # decoder
    decoder = conv_2d_transpose(encoder, 64, 3, strides=2, output_shape=[24, 76]) #output_shape = [32, 106])
    decoder = conv_2d(decoder, 64, 3, activation='relu')
    decoder = conv_2d(decoder, 64, 3, activation='relu')
    
    decoder = conv_2d_transpose(decoder, 32, 5, strides=2, output_shape=[48, 152]) #output_shape = [64, 212])
    decoder = conv_2d(decoder, 32, 5, activation='relu')
    decoder = conv_2d(decoder, 32, 5, activation='relu')

    decoder = conv_2d_transpose(decoder, 16, 7, strides=2, output_shape=[96, 304]) #output_shape = [128, 424])
    decoder = conv_2d(decoder, 16, 7, activation='relu')
    decoder = conv_2d(decoder, 16, 7, activation='relu')
    
    # decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 3, 1)

    def my_loss(y_pred, y_true):
        return tflearn.objectives.weak_cross_entropy_2d(y_pred, y_true, num_classes=3)
    
    def my_metric(y_pred, y_true):
        return tflean.metrics.Top_k(k=3)

    network = regression(decoder, 
                         optimizer='adam', 
                         #loss='mean_square', 
                         loss='categorical_crossentropy', 
                         #loss='weak_cross_entropy_2d', 
                         #loss=my_loss, 
                         #learning_rate=0.00005, 
                         #learning_rate=0.0005, 
                         #metric=my_metric
                        ) 

    return network

# segnet-like 
def build_segnet(network):
    HEIGHT = 240
    WIDTH  = 320

    #Pool1
    network_1 = conv_2d(network, 16, 3, activation='relu') #output 2x_downsampled
    network_1 = conv_2d(network_1, 16, 3, activation='relu') #output 2x_downsampled
    pool1 = max_pool_2d(network_1, 2)
    #Pool2
    network_2 = conv_2d(pool1, 32, 3, activation='relu') #output 4x_downsampled
    network_2 = conv_2d(network_2, 32, 3, activation='relu') #output 4x_downsampled
    pool2 = max_pool_2d(network_2, 2)
    #Pool3
    network_3 = conv_2d(pool2, 64, 3, activation='relu') #output 8x_downsampled
    network_3 = conv_2d(network_3, 64, 3, activation='relu') #output 8x_downsampled
    pool3 = max_pool_2d(network_3, 2)
    #Pool4
    network_4 = conv_2d(pool3, 128, 3, activation='relu') #output 16x_downsampled
    network_4 = conv_2d(network_4, 128, 3, activation='relu') #output 16x_downsampled
    pool4 = max_pool_2d(network_4, 2)

    # ----- decoder ----- 
    decoder = conv_2d_transpose(pool4, 128, 3, strides=4, output_shape=[HEIGHT//4, WIDTH//4, 128]) #  16x downsample to 4x downsample
    decoder = conv_2d(decoder, 128, 3, activation='relu')
    pool5 = conv_2d(decoder, 128, 3, activation='relu')
 
    decoder = conv_2d_transpose(pool3, 64, 3, strides=2, output_shape=[HEIGHT//4, WIDTH//4, 64]) # 8x downsample to 4x downsample
    decoder = conv_2d(decoder, 64, 3, activation='relu')
    pool6 = conv_2d(decoder, 64, 3, activation='relu')

    pool6=merge([pool6, pool5, pool2], mode='concat', axis=3) #merge all 4x downsampled layers

    decoder = conv_2d_transpose(pool6, 32, 3, strides=4, output_shape=[HEIGHT, WIDTH, 32])
    decoder = conv_2d(decoder, 32, 3, activation='relu')
    pool6 = conv_2d(decoder, 32, 3, activation='relu')
   
    decoder = conv_2d(pool6, 3, 1)
    network = tflearn.regression(decoder, optimizer='adam', loss='mean_square') 
    
    return network

# example of using upscore layer
def build_upscore(network, classes):
    network = conv_2d(network, 16, 5, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 32, 5, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 5, activation='relu')
    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = max_pool_2d(network, 2, strides=2)
    network = local_response_normalization(network)
    
    network = upscore_layer(network, num_classes=classes, shape=[4, 53, 16], kernel_size=8)

    network = regression(network, optimizer='adam', 
                         loss='weak_cross_entropy_2d', 
                         learning_rate=0.001)

    return network

# network visualizer
def build_visualizer(network, classes):
    network = conv_2d(network, 32, 3, activation='relu', strides=2) 
    layer1 = tf.nn.max_pool(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    network = conv_2d(network, 64, 3, activation='relu', strides=2) 
    network = conv_2d(network, 64, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') # 512
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.001)
    return network, layer1

# google net
def build_googlenet(network, classes):
    conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
    pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)
    conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

    # 3a
    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu', name='inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu', name='inception_3a_5_5_reduce')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name='inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool')
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    # 3b
    inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu', name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name='inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5, name='inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1, name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu', name='inception_3b_pool_1_1')
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat', axis=3, name='inception_3b_output')
    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

    # 4a
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5, activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1, name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')
    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

    # 4b
    inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
    inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
    inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
    inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5, activation='relu', name='inception_4b_5_5')
    inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1, name='inception_4b_pool')
    inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')
    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

    # 4c
    inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
    inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
    inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu', name='inception_4c_3_3')
    inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
    inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64, filter_size=5, activation='relu', name='inception_4c_5_5')
    inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')
    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3, name='inception_4c_output')

    # 4d
    inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
    inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
    inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
    inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
    inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5, activation='relu', name='inception_4d_5_5')
    inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1, name='inception_4d_pool')
    inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')
    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

    # 4e
    inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
    inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
    inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
    inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
    inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_4e_5_5')
    inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1, name='inception_4e_pool')
    inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')
    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=3, mode='concat')
    pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

    # 5a
    inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
    inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
    inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5a_5_5')
    inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1, name='inception_5a_pool')
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu', name='inception_5a_pool_1_1')
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3, mode='concat')

    # 5b
    inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
    inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
    inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu', name='inception_5b_3_3')
    inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
    inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5b_5_5')
    inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1, name='inception_5b_pool')
    inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')
    pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)

    # fc
    loss = fully_connected(pool5_7_7, classes, activation='softmax')
    network = regression(loss, optimizer='momentum', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)
    
    return network

# /////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////

def build_only_fc(network, classes, fc_units):
    network = conv_2d(network, 1, 3, activation='relu')
    network = conv_2d(network, 2)

    network = flatten(network)

    network = fully_connected(network, fc_units, activation='relu')
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)

    return network

# ////////// learning rate //////////
def buil_lrnet(network, classes, lr):
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', learning_rate=lr, 
                        loss='categorical_crossentropy', name='target')
    return network

# //////////////////// datasets ////////////////////
def build_kylberg(network, classes):
    network = conv_2d(network, 8, 3, activation='relu') 
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 16, 3, activation='relu')  
    network = max_pool_2d(network, 2)
    
    network = conv_2d(network, 24, 3, activation='relu')  
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 128, activation='relu') 
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)   
    return network

# digits
def build_digits(network, classes):
    network = conv_2d(network, 16, 3, activation='relu', regularizer=None)
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)

    network = conv_2d(network, 32, 3, activation='relu', regularizer=None)
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)
    
    network = conv_2d(network, 64, 3, activation='relu', regularizer=None)
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)
    
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', learning_rate=0.001, 
                        loss='categorical_crossentropy', name='target')
    return network

# fabric
def build_fabric(network, classes, lr=0.0001):
    network = conv_2d(network, 8, 5, activation='relu', strides=4)
    network = max_pool_2d(network, 2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam',  
                        loss='categorical_crossentropy', 
                        learning_rate=lr)    
    return network

# --- GTSD ---
def build_gtsd(network, classes):
    network = conv_2d(network, 32, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu') 
    network = conv_2d(network, 64, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = batch_normalization(network)

    network = flatten(network)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)
    return network

def build_gtsd_1layer(network, classes, nfilters):
    nfilters = int(nfilters)
    
    network = conv_2d(network, nfilters, 3, activation='relu')
    network = max_pool_2d(network, 2) 

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)
    return network

# 2 layer network
def build_gtsd_2layer(network, classes, nfilters):
    nfilters = int(nfilters)
    
    network = conv_2d(network, nfilters, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, nfilters*2, 3, activation='relu')
    network = max_pool_2d(network, 2)  

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)
    return network

# 3 layer network
def build_gtsd_3layer(network, classes, nfilters):
    nfilters = int(nfilters)
    
    network = conv_2d(network, nfilters, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, nfilters*2, 3, activation='relu')
    network = conv_2d(network, nfilters*2, 3, activation='relu')
    network = max_pool_2d(network, 2)  

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)
    return network

# 4 layer network
def build_gtsd_4layer(network, classes, nfilters):
    nfilters = int(nfilters)
    
    network = conv_2d(network, nfilters, 3, activation='relu')
    network = conv_2d(network, nfilters, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, nfilters*2, 3, activation='relu')
    network = conv_2d(network, nfilters*2, 3, activation='relu')
    network = max_pool_2d(network, 2)  

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)
    return network

# 5 layer network
def build_gtsd_5layer(network, classes, nfilters):
    nfilters = int(nfilters)
    
    network = conv_2d(network, nfilters, 3, activation='relu')
    network = conv_2d(network, nfilters, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, nfilters*2, 3, activation='relu')
    network = conv_2d(network, nfilters*2, 3, activation='relu')
    network = conv_2d(network, nfilters*2, 3, activation='relu')
    network = max_pool_2d(network, 2)  

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)
    
    return network

# net from https://navoshta.com/traffic-signs-classification/
def build_blog(network, classes):
    network1 = conv_2d(network, 32, 5, activation='relu')
    network1 = max_pool_2d(network1, 2)
    network1 = dropout(network1, 0.9)

    network2 = conv_2d(network1, 64, 5, activation='relu')
    network2 = max_pool_2d(network2, 2)
    network2 = dropout(network2, 0.8)

    network3 = conv_2d(network2, 128, 5, activation='relu')
    network3 = max_pool_2d(network3, 2)
    network3 = dropout(network3, 0.7)

    network1 = max_pool_2d(network1, 4)
    network2 = max_pool_2d(network2, 2)

    network = merge([network1, network2, network3], axis=3, mode='concat')

    network = flatten(network)

    network = fully_connected(network, 3584, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax') 

    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.0001)

    return network

# dynamic network VGG style
def build_dynamic(network, classes, depth, norm=0):
    depth = int(depth)
    
    fsize = 5
    nfc_layers = 1
    fc_units = 512

    b_norm = True 
    lr_norm = True and not b_norm

    # ////////// feature extraction //////////
    for i in range(depth):
        nfilters = 2 ** (i+5) # first = 16, 32
        for j in range(i+1):
            network = conv_2d(network, nfilters, fsize, activation='relu')
        
        #network = conv_2d(network, nfilters//2, 1, activation='relu')
        
        if(i < 3):
            network = max_pool_2d(network, 2)
        if(b_norm):
            network = batch_normalization(network)
        if(lr_norm):
            network = local_response_normalization(network)
    
    # ////////// classification //////////
    network = flatten(network)

    network = fully_connected(network, fc_units, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, fc_units//2, activation='relu')
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.0001)
    return network


def my_inception_module(network, nfilters):
    # 1x1 inception
    conv_1x1 = conv_2d(network, nfilters//2, 1, activation='relu')
    # 3x3 inception
    conv_3x3 = conv_2d(network, nfilters//2, 1, activation='relu')
    conv_3x3 = conv_2d(conv_3x3, nfilters, 3, activation='relu')
    # 5x5 inception
    conv_5x5 = conv_2d(network, nfilters//2, 1, activation='relu')
    conv_5x5 = conv_2d(conv_5x5, nfilters, 5, activation='relu')
    # overlaping max pooling (stride=1 -> no reduction)
    mp_3x3 = max_pool_2d(network, kernel_size=3, strides=1)
    mp_3x3 = conv_2d(mp_3x3, nfilters//2, 1, activation='relu')
    # merge all branches
    inception = merge([conv_1x1, conv_3x3, conv_5x5, mp_3x3], axis=3, mode='concat')

    return inception

def build_my_inception_net(network, classes, depth):
    depth = int(depth)
    filter_sizes = [32, 64, 64]
    filter_sizes = [32, 64, 128, 128]

    for i in range(depth):
        network = my_inception_module(network, filter_sizes[i])

        if(i < 3):
            network = max_pool_2d(network, 2)
    
    network = tflearn.global_avg_pool(network)
    network = dropout(network, 0.5)
    
    network = fully_connected(network, classes, activation='softmax')
    
    # momentum
    network = regression(network, optimizer='momentum', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)
    
    return network

def build_my_resnet(network, classes, depth):
    n = int(depth)
    network = tflearn.conv_2d(network, 16, 3, regularizer='L2', weight_decay=0.0001)
    network = tflearn.residual_block(network, n, 16)
    network = tflearn.residual_block(network, 1, 32, downsample=True)
    
    network = tflearn.residual_block(network, n-1, 32)
    network = tflearn.residual_block(network, 1, 64, downsample=True)
    
    network = tflearn.residual_block(network, n-1, 64)
    network = tflearn.batch_normalization(network)
    network = tflearn.activation(network, 'relu')
    network = tflearn.global_avg_pool(network)
    
    # Regression
    network = tflearn.fully_connected(network, classes, activation='softmax')
    mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    network = tflearn.regression(network, optimizer=mom, 
                            loss='categorical_crossentropy')
    
    return network

# ///////////////////////////////////////////////////////////////////////////////////////////
# network builder function
def build_network(name, network, classes, param=0):
    """
    Function to create a network topology/architecture.
    """
    print("[INFO] Loading network...")
    l1 = None
    
    if(name == "vgg16"):           network = build_vgg16(network, classes)
    if(name == "vgg16_ft"):        network = build_vgg16_ft(network, classes)
    elif(name == "myvgg"):         network = build_myvgg16(network, classes)
    elif(name == "mnist"):         network = build_mnist(network, classes)
    
    # ----- custom networks -----
    # ----- 1 layer -----
    elif(name == "1l_32f_5x5_fc512"):      network = build_1l_32f_5x5_fc512(network, classes)
    elif(name == "1l_32f_5x5_fc512_s2"):   network = build_1l_32f_5x5_fc512_s2(network, classes)
    elif(name == "1l_32f_5x5_fc412"):      network = build_1l_32f_5x5_fc412(network, classes)
    elif(name == "1l_32f_5x5_fc312"):      network = build_1l_32f_5x5_fc312(network, classes)
    elif(name == "1l_32f_5x5_fc212"):      network = build_1l_32f_5x5_fc212(network, classes)
    elif(name == "1l_32f_5x5_fc112"):      network = build_1l_32f_5x5_fc112(network, classes)
    
    elif(name == "1l_32f_5x5_fc50"):    network = build_1l_32f_5x5_fc50(network, classes)
    elif(name == "1l_32f_5x5_fc25"):    network = build_1l_32f_5x5_fc25(network, classes)
    elif(name == "1l_32f_5x5_fc10"):    network = build_1l_32f_5x5_fc10(network, classes)
    elif(name == "1l_32f_5x5_fc5"):     network = build_1l_32f_5x5_fc5(network, classes)

    elif(name == "1l_24f_5x5_fc50"):    network = build_1l_24f_5x5_fc50(network, classes)
    elif(name == "1l_16f_5x5_fc50"):    network = build_1l_16f_5x5_fc50(network, classes)
    elif(name == "1l_8f_5x5_fc50"):     network = build_1l_8f_5x5_fc50(network, classes)
    elif(name == "1l_4f_5x5_fc50"):     network = build_1l_4f_5x5_fc50(network, classes)
    elif(name == "1l_4f_5x5_fc50_nd"):  network = build_1l_4f_5x5_fc50_nd(network, classes)
    elif(name == "1l_2f_5x5_fc50"):     network = build_1l_2f_5x5_fc50(network, classes)
    elif(name == "1l_1f_5x5_fc50"):     network = build_1l_1f_5x5_fc50(network, classes)

    elif(name == "1l_8f_3x3_fc50"):     network = build_1l_8f_3x3_fc50(network, classes)
    elif(name == "1l_8f_7x7_fc50"):     network = build_1l_8f_7x7_fc50(network, classes)
    elif(name == "1l_8f_9x9_fc50"):     network = build_1l_8f_9x9_fc50(network, classes)
    elif(name == "1l_8f_11x11_fc50"):   network = build_1l_8f_11x11_fc50(network, classes)

    # ----- 2 layers -----
    elif(name == "2l_8f_16f_5x5_fc512"):      network = build_2l_8f_16f_5x5_fc512(network, classes)
    elif(name == "2l_8f_16f_5x5_fc256"):      network = build_2l_8f_16f_5x5_fc256(network, classes)
    elif(name == "2l_8f_16f_3x3_fc256"):      network = build_2l_8f_16f_3x3_fc256(network, classes)
    elif(name == "2l_8f_16f_5x5_fc256_ns"):   network = build_2l_8f_16f_5x5_fc256_ns(network, classes)
    elif(name == "2l_16f_32f_5x5_fc512"):     network = build_2l_16f_32f_5x5_fc512(network, classes)
    elif(name == "2l_32f_5x5_fc512"):         network = build_2l_32f_5x5_fc512(network, classes)
    elif(name == "2l_32f_5x5_fc512_ns"):      network = build_2l_32f_5x5_fc512_ns(network, classes)
    elif(name == "2l_32f_64f_3x3_fc512"):     network = build_2l_32f_64f_3x3_fc512(network, classes)
    elif(name == "2l_32f_64f_3x3_fc512_ns"):  network = build_2l_32f_64f_3x3_fc512_ns(network, classes)
    elif(name == "2l_32f_64f_5x5_fc512"):     network = build_2l_32f_64f_5x5_fc512(network, classes)
    elif(name == "2l_32f_64f_5x5_fc512_min"): network = build_2l_32f_64f_5x5_fc512_min(network, classes)
    elif(name == "2l_32f_64f_5x5_fc512_ns"):  network = build_2l_32f_64f_5x5_fc512_ns(network, classes)
    
    # ----- 3 layers -----
    elif(name == "3l_32f_3x3_fc512"):               network = build_3l_32f_3x3_fc512(network, classes)
    elif(name == "3l_32f_3x3_fc512_ns"):            network = build_3l_32f_3x3_fc512_ns(network, classes)
    elif(name == "3l_32f_5x5_fc512"):               network = build_3l_32f_5x5_fc512(network, classes)
    elif(name == "3l_32f_5x5_fc512_ns"):            network = build_3l_32f_5x5_fc512_ns(network, classes)
    elif(name == "3l_8f_16f_16f_5x5_fc256"):        network = build_3l_8f_16f_16f_5x5_fc256(network, classes)
    elif(name == "3l_16f_32f_32f_5x5_fc512"):       network = build_3l_16f_32f_32f_5x5_fc512(network, classes)
    elif(name == "3l_32f_64f_64f_3x3_fc512"):       network = build_3l_32f_64f_64f_3x3_fc512(network, classes)
    elif(name == "3l_32f_64f_64f_3x3_fc512_ns"):    network = build_3l_32f_64f_64f_3x3_fc512_ns(network, classes)
    elif(name == "3l_32f_64f_64f_533_fc512"):       network = build_3l_32f_64f_64f_533_fc512(network, classes)
    elif(name == "3l_32f_64f_64f_5x5_fc512"):       network = build_3l_32f_64f_64f_5x5_fc512(network, classes)
    elif(name == "3l_32f_64f_64f_5x5_fc256_fc512"): network = build_3l_32f_64f_64f_5x5_fc256_fc512(network, classes)
    elif(name == "3l_32f_64f_64f_5x5_fc512_ns"):    network = build_3l_32f_64f_64f_5x5_fc512_ns(network, classes)
    elif(name == "3l_32f_64f_128f_5x5_fc512"):      network = build_3l_32f_64f_128f_5x5_fc512(network, classes)

    # ----- 4 layers -----
    elif(name == "4l_32f_5x5_fc512"):                network = build_4l_32f_5x5_fc512(network, classes)
    elif(name == "4l_32f_5x5_fc512_ns"):             network = build_4l_32f_5x5_fc512_ns(network, classes)
    elif(name == "4l_16f_32f_48f_64f_5x5_fc512"):    network = build_4l_16f_32f_48f_64f_5x5_fc512(network, classes)
    elif(name == "4l_32f_32f_64f_64f_5x5_fc512_ns"): network = build_4l_32f_32f_64f_64f_5x5_fc512_ns(network, classes)

    # ----- other networks -----
    elif(name == "cifar10"):       network = build_cifar10(network, classes)
    elif(name == "cifar10_mod"):   network = build_cifar10_mod(network, classes)
    elif(name == "cifar10_mod2"):  network = build_cifar10_mod2(network, classes)    
    elif(name == "cifar10_valid"): network = build_cifar10_valid(network, classes)
    elif(name == "cifar10x2"):     network = build_cifar10_x2(network, classes)
    elif(name == "cifar10_half"):  network = build_cifar10_half(network, classes)
    elif(name == "mycifar"):       network = build_mycifar(network, classes)
    elif(name == "mycifarv2"):     network = build_mycifar_v2(network, classes)
    elif(name == "mycifarv3"):     network = build_mycifar_v3(network, classes)
    elif(name == "mycifarv4"):     network = build_mycifar_v4(network, classes)
    elif(name == "mycifarv5"):     network = build_mycifar_v5(network, classes)
    elif(name == "mycifarv6"):     network = build_mycifar_v6(network, classes)       
    elif(name == "resnet"):        network = build_resnet(network, classes, param)
    elif(name == "resnetv2"):      network = build_resnet_v2(network, classes, param)
    elif(name == "densenet"):      network = build_densenet(network, classes) 
    elif(name == "alexnet"):       network = build_alex(network, classes)
    elif(name == "myalex"):        network = build_myalex(network, classes)          
    elif(name == "nin"):           network = build_nin(network, classes) 
    elif(name == "highway"):       network = build_highway(network, classes) 
    elif(name == "rnn"):           network = build_rnn(network, classes)
    elif(name == "allcnn"):        network = build_all_cnn(network, classes)  
    elif(name == "dlib"):          network = build_dlib(network, classes)
    elif(name == "small"):         network = build_small_net(network, classes)   
    elif(name == "snet"):          network = build_snet(network, classes)  
    elif(name == "googlenet"):     network = build_googlenet(network, classes)
    
    elif(name == "merge"):         network = build_merge_test(network, classes)
    elif(name == "noconv"):        network = build_non_convolutional(network, classes)

    elif(name == "visual"):        network, l1 = build_visualizer(network, classes)
    elif(name == "custom"):        network = build_custom(network, classes)
    elif(name == "custom2"):       network = build_custom2(network, classes)
    elif(name == "custom3"):       network = build_custom3(network, classes)
    elif(name == "custom3.1"):     network = build_custom31(network, classes)
    elif(name == "custom3.2"):     network = build_custom32(network, classes)
    elif(name == "custom4"):       network = build_custom4(network, classes)
    elif(name == "lenet"):         network = build_lenet(network, classes)
    elif(name == "upscore"):       network = build_upscore(network, classes)
    
    elif(name == "fcn"):           network = build_fcn_all(network, classes)
    elif(name == "autoencoder"):   network = build_autoencoder(network, classes)
    elif(name == "segnet"):        network = build_segnet(network)

    # ////////////////////////////// dissertation //////////////////////////////
    elif(name == "lrnet"):         network = buil_lrnet(network, classes, param)

    elif(name == "kylberg"):       network = build_kylberg(network, classes)
    elif(name == "digits"):        network = build_digits(network, classes)
    elif(name == "fabric"):        network = build_fabric(network, classes, param)
    elif(name == "gtsd"):          network = build_gtsd(network, classes)
    elif(name == "gtsd_1l"):       network = build_gtsd_1layer(network, classes, param)
    elif(name == "gtsd_2l"):       network = build_gtsd_2layer(network, classes, param)
    elif(name == "gtsd_3l"):       network = build_gtsd_3layer(network, classes, param)
    elif(name == "gtsd_4l"):       network = build_gtsd_4layer(network, classes, param)
    elif(name == "gtsd_5l"):       network = build_gtsd_5layer(network, classes, param)

    elif(name == "dynamic"):       network = build_dynamic(network, classes, param)
    elif(name == "inception"):     network = build_my_inception_net(network, classes, param)
    elif(name == "blog"):          network = build_blog(network, classes)
    
    # //////////////////////////////////////////////////////////////////////////
    elif(name == "tyres"):         network = build_tyres(network, classes)

    else: sys.exit(colored("ERROR: Unknown architecture!", "red"))

    print("[INFO] Architecture:", name)
    print("[INFO] Network loaded!\n")

    return network, l1