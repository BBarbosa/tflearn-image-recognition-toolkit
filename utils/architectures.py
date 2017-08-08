from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import sys
import numpy as np
from tflearn.layers.core import input_data,dropout,fully_connected,flatten
from tflearn.layers.conv import conv_2d,max_pool_2d,highway_conv_2d,avg_pool_2d,upsample_2d,upscore_layer,conv_2d_transpose
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization,batch_normalization
from tflearn.layers.merge_ops import merge
from colorama import init
from termcolor import colored

# init colored print
init()

# vgg16 (heavy) 
def build_vgg16(network,classes):
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

# vgg16 (light) 
def build_myvgg16(network,classes):
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)
    return network

# mnist 
def build_mnist(network,classes):
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
def build_cifar10(network,classes):
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

# cifar10 valid
def build_cifar10_valid(network,classes):
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
def build_cifar10_x2(network,classes):
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
def build_cifar10_x05(network,classes):
    network = conv_2d(network, 16, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 3, activation='relu') 
    network = conv_2d(network, 32, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 256, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.001)
    return network

# mycifar (fixed)
def build_mycifar(network,classes):
    network = conv_2d(network, 32, 11, activation='relu',strides=2) 
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
def build_mycifar_v2(network,classes):
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
    
    sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=125) # 0.005

    network = regression(network, optimizer=sgd,
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.01)    # 0.00001
    return network

# mycifarv3
def build_mycifar_v3(network,classes):
    network = conv_2d(network, 32, 3, activation='relu',strides=2) 
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
def build_mycifar_v4(network,classes):
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
def build_mycifar_v5(network,classes):
    network = conv_2d(network, 32, 11, activation='relu',strides=4) 
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
def build_mycifar_v6(network,classes):
    network = conv_2d(network, 32, 11, activation='relu',strides=4) 
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
def build_1l_32f_5x5_fc512(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network

def build_1l_32f_5x5_fc512_s2(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)   
    return network

def build_1l_32f_5x5_fc412(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 412, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_32f_5x5_fc312(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 312, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc212(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc112(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 112, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc50(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc25(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 25, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc10(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 10, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_32f_5x5_fc5(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 5, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_32f_5x5_fc4(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 4, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_32f_5x5_fc2(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 2, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_32f_5x5_fc1(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 1, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


def build_1l_24f_5x5_fc50(network,classes):
    network = conv_2d(network, 24, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network,50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_16f_5x5_fc50(network,classes):
    network = conv_2d(network, 16, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_8f_5x5_fc50(network,classes):
    network = conv_2d(network, 8, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_4f_5x5_fc50(network,classes):
    network = conv_2d(network, 4, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_8f_3x3_fc50(network,classes):
    network = conv_2d(network, 8, 3, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_8f_7x7_fc50(network,classes):
    network = conv_2d(network, 8, 7, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_8f_9x9_fc50(network,classes):
    network = conv_2d(network, 8, 9, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network
 
def build_1l_8f_11x11_fc50(network,classes):
    network = conv_2d(network, 8, 11, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_4f_5x5_fc50_nd(network,classes):
    network = conv_2d(network, 4, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    # same as v114 but without dropout
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_2f_5x5_fc50(network,classes):
    network = conv_2d(network, 2, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

def build_1l_1f_5x5_fc50(network,classes):
    network = conv_2d(network, 1, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 50, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# 2 conv layers network ------------------------------------------------------------------
def build_2l_32f_5x5_fc512(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 5, activation='relu', strides=2)  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network


def build_2l_32f_5x5_fc512_ns(network,classes):
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

def build_2l_32f_64f_3x3_fc512(network,classes):
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

def build_2l_32f_64f_3x3_fc512_ns(network,classes):
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

def build_2l_32f_64f_5x5_fc512(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu', strides=2)  
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.0001)    
    return network


def build_2l_32f_64f_5x5_fc512_ns(network,classes):
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
def build_3l_32f_3x3_fc512(network,classes):
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


def build_3l_32f_3x3_fc512_ns(network,classes):
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

def build_3l_32f_5x5_fc512(network,classes):
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


def build_3l_32f_5x5_fc512_ns(network,classes):
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

def build_3l_32f_64f_64f_3x3_fc512(network,classes):
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
                        learning_rate=0.0001)    
    return network


def build_3l_32f_64f_64f_3x3_fc512_ns(network,classes):
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
                        learning_rate=0.0001)    
    return network

def build_3l_32f_64f_64f_5x5_fc512(network,classes):
    network = conv_2d(network, 32, 5, activation='relu',strides=2) 
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

def build_3l_32f_64f_64f_5x5_fc256_fc512(network,classes):
    network = conv_2d(network, 32, 5, activation='relu',strides=2) 
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

def build_3l_32f_64f_64f_5x5_fc512_ns(network,classes):
    network = conv_2d(network, 32, 5, activation='relu') 
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
                        learning_rate=0.0001)    
    return network

def build_3l_32f_64f_128f_5x5_fc512(network,classes):
    network = conv_2d(network, 32, 5, activation='relu',strides=2) 
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
def build_4l_32f_5x5_fc512(network,classes):
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

def build_4l_32f_5x5_fc512_ns(network,classes):
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

def build_4l_16f_32f_48f_64f_5x5_fc512(network,classes):
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

def build_4l_32f_32f_64f_64f_5x5_fc512_ns(network,classes):
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
def build_all_cnn(network,classes):
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
def build_alex(network,classes):
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
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
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='momentum',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)
    return network 

# my alexnet
def build_myalex(network,classes):
    network = conv_2d(network, 96, 11, strides=2, activation='relu')
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
                        learning_rate=0.001)
    return network 


# network in network (error)
def build_nin(network,classes):
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
def build_highway(network,classes):
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
def build_rnn(network,classes):

    network = tflearn.lstm(network, 128, return_seq=True)
    network = tflearn.lstm(network, 128)
    network = tflearn.fully_connected(network, classes, activation='softmax')
    
    network = tflearn.regression(network, optimizer='adam',
                            loss='categorical_crossentropy', name="output1")
    return network

# resnet (error)
def build_resnet(network,classes):
    n=5
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

# dlib (used on mnist)
def build_dlib(network,classes):
    network = conv_2d(network, 6, 5, activation='relu', padding='same',strides=2) 
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

# small net
def build_snet(network,classes):
    network = conv_2d(network, 8, 5, activation='relu', strides=4)
    network = max_pool_2d(network,4)

    network = fully_connected(network, 128, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# merge test
def build_merge_test(network,in2,classes):
    network = conv_2d(network, 10, 2, activation='relu', strides=4)
    network = max_pool_2d(network,2)

    network = fully_connected(network, 128, activation='relu')
    network = merge([network,in2],mode='concat')
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# costum net
def build_custom(network,classes):
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
def build_custom2(network,classes):
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
def build_custom3(network,classes):
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

# lenet net
def build_lenet(network,classes):
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
def build_non_convolutional(network,classes):
    network = max_pool_2d(network, 20)
    network = conv_2d(network, 10, 3, activation='relu')

    network = fully_connected(network, 256,activation='relu')
    network = fully_connected(network,classes,activation='softmax') 

    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy',    
                        learning_rate=0.0001)   

    return network

# autoencoder example
def build_autoencoder(network,classes):                            
    encoder = conv_2d(network, 16, 5, activation='relu') 
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 32, 5, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 64, 5, activation='relu')
    encoder = max_pool_2d(encoder, 2)

    #encoder = conv_2d(encoder, 64, 5, activation='relu')
    #encoder = max_pool_2d(encoder, 2)
    
    #decoder = conv_2d(encoder, 64, 5, activation='relu')
    #decoder = upsample_2d(decoder, 2)

    decoder = conv_2d(encoder, 64, 5, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 32, 5, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 16, 5, activation='relu')
    decoder = upsample_2d(decoder, 2)
    
    decoder = conv_2d(decoder, 3, 5) # NOTE: always check color space

    # transpose (when use stride)
    #decoder = conv_2d_transpose(encoder, 64, 5, [16,53], activation='relu')
    #decoder = upsample_2d(decoder, 2)
    #decoder = conv_2d_transpose(decoder, 32, 5, [32,106], activation='relu')
    #decoder = upsample_2d(decoder, 2)
    #decoder = conv_2d_transpose(decoder, 16, 5, [64,212], activation='relu')
    #decoder = upsample_2d(decoder, 2)

    #decoder = conv_2d_transpose(decoder, 8, 5, [64,64], activation='relu')
    #decoder = upsample_2d(decoder, 2)
    
    #decoder = conv_2d_transpose(decoder, 3, 5, [128,424])

    # FC -> Convolutional
    # network = conv_2d(network, 512, 8, activation='relu')  

    network = regression(decoder, 
                         optimizer='adam',
                         loss='mean_square',    
                         learning_rate=0.0001,
                         metric=None)  

    return network

# example of using upscore layer
def build_upscore(network,classes):
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

    network = regression(network,optimizer='adam',
                         loss='weak_cross_entropy_2d',
                         learning_rate=0.001)

    return network

# network visualizer
def build_visualizer(network,classes):
    network = conv_2d(network, 32, 3, activation='relu', strides=2) 
    layer1 = tf.nn.max_pool(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    network = conv_2d(network, 64, 3, activation='relu',strides=2) 
    network = conv_2d(network, 64, 3, activation='relu') 
    network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') # 512
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.001)
    return network,layer1

# network builder function
def build_network(name,network,classes):
    """
    Function to create a network topology/architecture.
    """
    print("Loading network...")
    l1 = None
    
    if(name == "vgg16"):           network = build_vgg16(network,classes)
    elif(name == "myvgg"):         network = build_myvgg16(network,classes)
    elif(name == "mnist"):         network = build_mnist(network,classes)
    
    # ----- custom networks -----
    # ----- 1 layer -----
    elif(name == "1l_32f_5x5_fc512"):      network = build_1l_32f_5x5_fc512(network,classes)
    elif(name == "1l_32f_5x5_fc512_s2"):   network = build_1l_32f_5x5_fc512_s2(network,classes)
    elif(name == "1l_32f_5x5_fc412"):      network = build_1l_32f_5x5_fc412(network,classes)
    elif(name == "1l_32f_5x5_fc312"):      network = build_1l_32f_5x5_fc312(network,classes)
    elif(name == "1l_32f_5x5_fc212"):      network = build_1l_32f_5x5_fc212(network,classes)
    elif(name == "1l_32f_5x5_fc112"):      network = build_1l_32f_5x5_fc112(network,classes)
    
    elif(name == "1l_32f_5x5_fc50"):    network = build_1l_32f_5x5_fc50(network,classes)
    elif(name == "1l_32f_5x5_fc25"):    network = build_1l_32f_5x5_fc25(network,classes)
    elif(name == "1l_32f_5x5_fc10"):    network = build_1l_32f_5x5_fc10(network,classes)
    elif(name == "1l_32f_5x5_fc5"):     network = build_1l_32f_5x5_fc5(network,classes)

    elif(name == "1l_24f_5x5_fc50"):    network = build_1l_24f_5x5_fc50(network,classes)
    elif(name == "1l_16f_5x5_fc50"):    network = build_1l_16f_5x5_fc50(network,classes)
    elif(name == "1l_8f_5x5_fc50"):     network = build_1l_8f_5x5_fc50(network,classes)
    elif(name == "1l_4f_5x5_fc50"):     network = build_1l_4f_5x5_fc50(network,classes)
    elif(name == "1l_4f_5x5_fc50_nd"):  network = build_1l_4f_5x5_fc50_nd(network,classes)
    elif(name == "1l_2f_5x5_fc50"):     network = build_1l_2f_5x5_fc50(network,classes)
    elif(name == "1l_1f_5x5_fc50"):     network = build_1l_1f_5x5_fc50(network,classes)

    elif(name == "1l_8f_3x3_fc50"):     network = build_1l_8f_3x3_fc50(network,classes)
    elif(name == "1l_8f_7x7_fc50"):     network = build_1l_8f_7x7_fc50(network,classes)
    elif(name == "1l_8f_9x9_fc50"):     network = build_1l_8f_9x9_fc50(network,classes)
    elif(name == "1l_8f_11x11_fc50"):   network = build_1l_8f_11x11_fc50(network,classes)

    # ----- 2 layers -----
    elif(name == "2l_32f_5x5_fc512"):         network = build_2l_32f_5x5_fc512(network,classes)
    elif(name == "2l_32f_5x5_fc512_ns"):      network = build_2l_32f_5x5_fc512_ns(network,classes)
    elif(name == "2l_32f_64f_3x3_fc512"):     network = build_2l_32f_64f_3x3_fc512(network,classes)
    elif(name == "2l_32f_64f_3x3_fc512_ns"):  network = build_2l_32f_64f_3x3_fc512_ns(network,classes)
    elif(name == "2l_32f_64f_5x5_fc512"):     network = build_2l_32f_64f_5x5_fc512(network,classes)
    elif(name == "2l_32f_64f_5x5_fc512_ns"):  network = build_2l_32f_64f_5x5_fc512_ns(network,classes)
    
    # ----- 3 layers -----
    elif(name == "3l_32f_3x3_fc512"):               network = build_3l_32f_3x3_fc512(network,classes)
    elif(name == "3l_32f_3x3_fc512_ns"):            network = build_3l_32f_3x3_fc512_ns(network,classes)
    elif(name == "3l_32f_5x5_fc512"):               network = build_3l_32f_5x5_fc512(network,classes)
    elif(name == "3l_32f_5x5_fc512_ns"):            network = build_3l_32f_5x5_fc512_ns(network,classes)
    elif(name == "3l_32f_64f_64f_3x3_fc512"):       network = build_3l_32f_64f_64f_3x3_fc512(network,classes)
    elif(name == "3l_32f_64f_64f_3x3_fc512_ns"):    network = build_3l_32f_64f_64f_3x3_fc512_ns(network,classes)
    elif(name == "3l_32f_64f_64f_5x5_fc512"):       network = build_3l_32f_64f_64f_5x5_fc512(network,classes)
    elif(name == "3l_32f_64f_64f_5x5_fc256_fc512"): network = build_3l_32f_64f_64f_5x5_fc256_fc512(network,classes)
    elif(name == "3l_32f_64f_64f_5x5_fc512_ns"):    network = build_3l_32f_64f_64f_5x5_fc512_ns(network,classes)
    elif(name == "3l_32f_64f_128f_5x5_fc512"):      network = build_3l_32f_64f_128f_5x5_fc512(network,classes)

    # ----- 4 layers -----
    elif(name == "4l_32f_5x5_fc512"):                network = build_4l_32f_5x5_fc512(network,classes)
    elif(name == "4l_32f_5x5_fc512_ns"):             network = build_4l_32f_5x5_fc512_ns(network,classes)
    elif(name == "4l_16f_32f_48f_64f_5x5_fc512"):    network = build_4l_16f_32f_48f_64f_5x5_fc512(network,classes)
    elif(name == "4l_32f_32f_64f_64f_5x5_fc512_ns"): network = build_4l_32f_32f_64f_64f_5x5_fc512_ns(network,classes)

    # ----- other networks -----
    elif(name == "cifar10"):       network = build_cifar10(network,classes)
    elif(name == "cifar10_valid"): network = build_cifar10_valid(network,classes)
    elif(name == "cifar10x2"):     network = build_cifar10_x2(network,classes)
    elif(name == "cifar10x0.5"):   network = build_cifar10_x05(network,classes)
    elif(name == "mycifar"):       network = build_mycifar(network,classes)
    elif(name == "mycifarv2"):     network = build_mycifar_v2(network,classes)
    elif(name == "mycifarv3"):     network = build_mycifar_v3(network,classes)
    elif(name == "mycifarv4"):     network = build_mycifar_v4(network,classes)
    elif(name == "mycifarv5"):     network = build_mycifar_v5(network,classes)
    elif(name == "mycifarv6"):     network = build_mycifar_v6(network,classes)       
    elif(name == "resnet"):        network = build_resnet(network,classes) 
    elif(name == "alexnet"):       network = build_alex(network,classes)
    elif(name == "myalex"):        network = build_myalex(network,classes)          
    elif(name == "nin"):           network = build_nin(network,classes) 
    elif(name == "highway"):       network = build_highway(network,classes) 
    elif(name == "rnn"):           network = build_rnn(network,classes)
    elif(name == "allcnn"):        network = build_all_cnn(network,classes)  
    elif(name == "dlib"):          network = build_dlib(network,classes)   
    elif(name == "snet"):          network = build_snet(network,classes)   
    
    elif(name == "merge"):         network = build_merge_test(network,classes)
    elif(name == "noconv"):        network = build_non_convolutional(network, classes)

    elif(name == "autoencoder"):   network = build_autoencoder(network,classes)
    elif(name == "visual"):        network,l1 = build_visualizer(network,classes)
    elif(name == "custom"):        network = build_custom(network,classes)
    elif(name == "custom2"):       network = build_custom2(network,classes)
    elif(name == "custom3"):       network = build_custom3(network,classes)
    elif(name == "lenet"):         network = build_lenet(network,classes)
    elif(name == "upscore"):       network = build_upscore(network,classes)
    
    else: sys.exit(colored("ERROR: Unknown architecture!","red"))

    print("\tArchitecture: ",name)
    print("Network loaded!\n")

    return network,l1