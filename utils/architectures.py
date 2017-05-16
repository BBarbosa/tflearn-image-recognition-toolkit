from __future__ import division, print_function, absolute_import

import tflearn
import sys

from tflearn.layers.core import input_data, dropout, fully_connected,flatten
from tflearn.layers.conv import conv_2d, max_pool_2d,highway_conv_2d,avg_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization,batch_normalization

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

# mynet
def build_mynet(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v11
def build_mynet_v11(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 412, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v12
def build_mynet_v12(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 312, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v13
def build_mynet_v13(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v14
def build_mynet_v14(network,classes):
    network = conv_2d(network, 32, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 112, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v15 -----------------------
def build_mynet_v15(network,classes):
    network = conv_2d(network, 24, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v16
def build_mynet_v16(network,classes):
    network = conv_2d(network, 16, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v17
def build_mynet_v17(network,classes):
    network = conv_2d(network, 8, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v18
def build_mynet_v18(network,classes):
    network = conv_2d(network, 4, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v19 -----------------------------------------------
def build_mynet_v19(network,classes):
    network = conv_2d(network, 8, 3, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v111
def build_mynet_v111(network,classes):
    network = conv_2d(network, 8, 7, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v112
def build_mynet_v112(network,classes):
    network = conv_2d(network, 8, 9, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v113
def build_mynet_v113(network,classes):
    network = conv_2d(network, 8, 11, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 212, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v114
def build_mynet_v114(network,classes):
    network = conv_2d(network, 4, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 48, activation='relu') 
    network = dropout(network, 0.75) 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v115
def build_mynet_v115(network,classes):
    network = conv_2d(network, 4, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 48, activation='relu') 
    # same as v114 but without dropout
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v116
def build_mynet_v116(network,classes):
    network = conv_2d(network, 1, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 48, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v117
def build_mynet_v117(network,classes):
    network = conv_2d(network, 2, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2) 
    
    network = fully_connected(network, 48, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# mynet_v2 ------------------------------
def build_mynet_v2(network,classes):
    network = conv_2d(network, 32, 3, activation='relu', strides=4) 
    #network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 7, activation='relu', strides=4) 
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

# mynet_v3
def build_mynet_v3(network,classes):
    network = conv_2d(network, 32, 3, activation='relu', strides=4) 
    #network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 7, activation='relu', strides=4) 
    network = conv_2d(network, 96, 5, activation='relu') 
    #network = max_pool_2d(network, 2)
    
    network = fully_connected(network, 512, activation='relu') 
    network = dropout(network, 0.5) 
    network = fully_connected(network, classes, activation='softmax')

    #sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.97, decay_step=41) # 0.005

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network


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
    network = conv_2d(network, 6, 5, activation='relu', strides=2) 
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 16, 5, activation='relu', strides=2)
    network = max_pool_2d(network, 2) 

    network = fully_connected(network, 120, activation='relu') 
    network = fully_connected(network, 84, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')
    
    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.001)    # 0.0001
    return network

# small net
def build_snet(network,classes):
    network = conv_2d(network, 8, 5, activation='relu', strides=4)
    network = max_pool_2d(network,2)

    network = fully_connected(network, 128, activation='relu') 
    network = fully_connected(network, classes, activation='softmax')

    network = regression(network, optimizer='adam', # 'adam',
                        loss='categorical_crossentropy', # loss='categorical_crossentropy'
                        learning_rate=0.0001)    # 0.00005
    return network

# network builder function
def build_network(name,network,classes):
    """
    Function to create a network.
    """
    print("Loading network...")
    
    if(name == "vgg16"):           network = build_vgg16(network,classes)
    elif(name == "myvgg"):         network = build_myvgg16(network,classes)
    elif(name == "mnist"):         network = build_mnist(network,classes)
    
    elif(name == "mynet"):         network = build_mynet(network,classes)
    elif(name == "mynet_v11"):     network = build_mynet_v11(network,classes)
    elif(name == "mynet_v12"):     network = build_mynet_v12(network,classes)
    elif(name == "mynet_v13"):     network = build_mynet_v13(network,classes)
    elif(name == "mynet_v14"):     network = build_mynet_v14(network,classes)

    elif(name == "mynet_v15"):     network = build_mynet_v15(network,classes)
    elif(name == "mynet_v16"):     network = build_mynet_v16(network,classes)
    elif(name == "mynet_v17"):     network = build_mynet_v17(network,classes)
    elif(name == "mynet_v18"):     network = build_mynet_v18(network,classes)

    elif(name == "mynet_v19"):     network = build_mynet_v19(network,classes)
    elif(name == "mynet_v111"):    network = build_mynet_v111(network,classes)
    elif(name == "mynet_v112"):    network = build_mynet_v112(network,classes)
    elif(name == "mynet_v113"):    network = build_mynet_v113(network,classes)

    elif(name == "mynet_v114"):    network = build_mynet_v114(network,classes)
    elif(name == "mynet_v115"):    network = build_mynet_v115(network,classes)
    elif(name == "mynet_v116"):    network = build_mynet_v116(network,classes)
    elif(name == "mynet_v117"):    network = build_mynet_v117(network,classes)
    
    elif(name == "mynetv2"):       network = build_mynet_v2(network,classes)
    elif(name == "mynetv3"):       network = build_mynet_v3(network,classes)
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
    else: sys.exit("ERROR: Unknown architecture!")

    print("\tArchitecture: ",name)
    print("Network loaded!\n")

    return network