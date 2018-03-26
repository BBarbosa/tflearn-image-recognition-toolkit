"""
Stand-alone python script for training and testing a convolutional 
neural network for pixel-wise image segmentation

Uses Tensorflow and TFLearn

Author: bbarbosa
Date: 20-05-2017
"""

from __future__ import division, print_function, absolute_import

import os
import sys
import cv2
import time
import platform
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn
import tensorflow as tf

from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_dirs_to_samples
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d, upscore_layer, conv_2d_transpose, residual_block

from PIL import Image
from colorama import init
from termcolor import colored

# init colored print
init()

# argument parser~
custom_formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=2000)
parser = argparse.ArgumentParser(description="Automatic image segmentation using Deep Learning.", 
                                 prefix_chars='-',
                                 formatter_class=custom_formatter_class) 
# required arguments
parser.add_argument("--arch", required=True, help="<REQUIRED> architecture name", type=str)
parser.add_argument("--model_name", required=True, help="<REQUIRED> run identifier (id) / model's path", type=str)
# optional arguments
parser.add_argument("--data_dir", required=False, help="directory to the training data", type=str)
parser.add_argument("--bsize", required=False, default=2, help="training batch size (default=2)", type=int)
parser.add_argument("--test_dir", required=False, help="path to testing images", type=str)
parser.add_argument("--height", required=False, help="images height (default=64)", default=64, type=int)
parser.add_argument("--width", required=False, help="images width (default=64)", default=64, type=int)
parser.add_argument("--gray", required=False, help="convert images to grayscale (default=False)", default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--video", required=False, help="use video capture device/video (default=0)", default=0)
parser.add_argument("--save", required=False, help="save output image (default=False)", default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--freeze", required=False, help="flag to freeze model (default=False)", default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--show", required=False, help="image show level (0-low; 1-medium; 2-high) (default=0)", default=0, type=int)

# parse arguments
args = parser.parse_args()

# print args
print(args, "\n")

# images properties
HEIGHT = args.height
WIDTH  = args.width

if(args.gray): 
    CHANNELS = 1 
else: 
    CHANNELS = 3

# load dataset and get image dimensions
if(args.data_dir is not None and args.bsize is not None):
    print("[INFO] Training image folder:", args.data_dir)
    X, _ = image_dirs_to_samples(args.data_dir, resize=(WIDTH, HEIGHT), convert_gray=args.gray, 
                                 filetypes=[".png", ".jpg", ".bmp"])
    
    #Xgt, _ = image_dirs_to_samples(args.data_dir, resize=(WIDTH, HEIGHT), convert_gray=args.gray, 
    #                            filetypes=[".png", ".jpg", ".bmp"])

    # images / ground truth split
    split = len(X) // 2
    Xim = X[:split]
    OXgt = X[split:] 

    OXgt = np.array(OXgt)
    OXgt = np.reshape(OXgt, (-1, HEIGHT, WIDTH, 1))
    Xgt = []

    for index, elem in enumerate(OXgt):
        Xgt.append(cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB))
    
    Xgt = np.array(Xgt)
    Xgt = np.reshape(Xgt, (-1, HEIGHT, WIDTH, 3))
    
    print("")
    print("[INFO] Images: ", len(Xim), Xim[0].shape)
    print("[INFO] Ground: ", len(Xgt), Xgt[0].shape, "\n")
else:
    print("[INFO] Training images not set")
    print("[INFO] Going for testing mode\n")

# Real-time data preprocessing  
img_prep = ImagePreprocessing()
img_prep.add_samplewise_zero_center()
img_prep.add_samplewise_stdnorm()    

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=10.)

# computational resources definition (made changes on TFLearn's config.py)
tflearn.init_graph(num_cores=8, allow_growth=True)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS],    # shape=[None, IMAGE, IMAGE] for RNN
                     data_preprocessing=None,                  # NOTE: always check PP
                     data_augmentation=None)                   # NOTE: always check DA

# ////////////////////////////////////////////////////
#        Custom metrics & loss functions
# ////////////////////////////////////////////////////
def custom_loss(y_pred, y_true):
    return tflearn.objectives.weak_cross_entropy_2d(y_pred, y_true, num_classes=4)

def custom_metric(y_pred, t_true):
    return tflearn.metrics.top_k_op(y_pred, t_true, k=3)

def compute_iou(y_pred_batch, y_true_batch):
    iterator = range(len(y_true_batch))
    return np.mean(np.asarray([pixel_accuracy(y_pred_batch[i], y_true_batch[i]) for i in iterator])) 

def pixel_accuracy(y_pred, y_true):
    img_rows = img_cols = 256
    y_pred = np.reshape(y_pred,[CHANNELS, img_rows, img_cols])
    y_true = np.reshape(y_true,[CHANNELS, img_rows, img_cols])
    y_pred = y_pred * (y_true>0)

    return 1.0 * np.sum((y_pred == y_true) * (y_true > 0)) /  np.sum(y_true > 0)

def iou(prediction, target, input):
    prediction_ = tf.cast(prediction, tf.float32)
    targets_ = tf.cast(target, tf.float32)

def custom_l1_norm(prediction, target, inputs):
    return tf.reduce_sum(tf.abs(prediction - target), name='l1')


# ////////////////////////////////////////////////////
#         Network architectures definition
# ////////////////////////////////////////////////////
# autoencoder example
def build_autoencoder(network): 
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
    decoder = conv_2d_transpose(encoder, 64, 3, strides=2, output_shape=[HEIGHT//4, WIDTH//4, 64])
    decoder = conv_2d(decoder, 64, 3, activation='relu')
    decoder = conv_2d(decoder, 64, 3, activation='relu')
    
    decoder = conv_2d_transpose(decoder, 32, 5, strides=2, output_shape=[HEIGHT//2, WIDTH//2, 32])
    decoder = conv_2d(decoder, 32, 5, activation='relu')
    decoder = conv_2d(decoder, 32, 5, activation='relu')

    decoder = conv_2d_transpose(decoder, 16, 7, strides=2, output_shape=[HEIGHT, WIDTH, 16])
    decoder = conv_2d(decoder, 16, 7, activation='relu')
    decoder = conv_2d(decoder, 16, 7, activation='relu')
    
    # decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, CHANNELS, 1)

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

# fully-convolutional network
def build_fcn_all(network):
    #Pool1
    conv1 = conv_2d(network, 8, 7, activation='relu')
    pool1 = max_pool_2d(conv1, 2)
    #Pool2
    conv2 = conv_2d(pool1, 16, 5, activation='relu')
    pool2 = max_pool_2d(conv2, 2)
    #Pool3
    conv3 = conv_2d(pool2, 32, 5, activation='relu')
    pool3 = max_pool_2d(conv3, 2)                       # output 8x_downsampled
    #Pool4
    conv4 = conv_2d(pool3, 64, 3, activation='relu') 
    pool4 = max_pool_2d(conv4, 2)                       # output 16x_downsampled

    #start FCN-32s -----------------------------------  
    #Pool5
    conv5 = conv_2d(pool4, 128, 3, activation='relu')
    pool5 = max_pool_2d(conv5, 2)
    #Conv6-7 
    conv6 = conv_2d(pool5, 128, 3, activation='relu')
    conv7 = conv_2d(conv6, 128, 3, activation='relu')   # output 32x_downsampled
    #end FCN-32s -----------------------------------

    ##start FCN-16s -----------------------------------
    #network_32_UP2 = upsample_2d(network_32, 2)
    #network_16 = merge([network_32_UP2, network_4], mode='concat', axis=3)
    #network_16 = conv_2d(network_16, 3, 128, activation='relu') # output 16x_downsampled
    ##end FCN-16s -----------------------------------

    ##start FCN-8s -----------------------------------
    #network_32_UP4 = upsample_2d(network_32, 4)
    #network_16_UP2  = upsample_2d(network_16, 2)
    #network_3_UP8   = upsample_2d(network_3, 8)
    pool4_x2 = upsample_2d(pool4, 2)
    conv7_x4 = upsample_2d(conv7, 4)
    #network_8 = merge([network_32_UP4, network_4_UP2, network_3], mode='concat', axis=3)
    fcn_8s = merge([pool3, pool4_x2, conv7_x4], mode='concat', axis=3)
    fcn_8s = conv_2d(fcn_8s, 3, 1, activation='relu')
    ##end FCN-8s -----------------------------------
    
    out = conv_2d(fcn_8s, CHANNELS, 1, activation='relu')
    out = upsample_2d(out, 8)
    #network_8 = upscore_layer(network_8, num_classes=3, kernel_size=2, strides=8, shape=[384, 1216, 3])
    
    network = tflearn.regression(out, 
                                 loss='mean_square', 
                                 #loss='weak_cross_entropy_2d', 
                                )

    return network

# segnet-like 
def build_segnet(network):
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
    network_3 = conv_2d(pool3, 128, 3, activation='relu') #output 16x_downsampled
    network_3 = conv_2d(network_3, 128, 3, activation='relu') #output 16x_downsampled
    pool4 = max_pool_2d(network_3, 2)

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
   
    decoder = conv_2d(pool6, CHANNELS, 1)
    network = tflearn.regression(decoder, optimizer='adam', loss='mean_square') 
    
    return network

# segnet-like half filters
def build_segnet_half(network):
    #Pool1
    network_1 = conv_2d(network, 8, 3, activation='relu') #output 2x_downsampled
    network_1 = conv_2d(network_1, 8, 3, activation='relu') #output 2x_downsampled
    pool1 = max_pool_2d(network_1, 2)
    #Pool2
    network_2 = conv_2d(pool1, 16, 3, activation='relu') #output 4x_downsampled
    network_2 = conv_2d(network_2, 16, 3, activation='relu') #output 4x_downsampled
    pool2 = max_pool_2d(network_2, 2)
    #Pool3
    network_3 = conv_2d(pool2, 32, 3, activation='relu') #output 8x_downsampled
    network_3 = conv_2d(network_3, 32, 3, activation='relu') #output 8x_downsampled
    pool3 = max_pool_2d(network_3, 2)

    #Pool4
    network_3 = conv_2d(pool3, 64, 3, activation='relu') #output 16x_downsampled
    network_3 = conv_2d(network_3, 64, 3, activation='relu') #output 16x_downsampled
    pool4 = max_pool_2d(network_3, 2)

    # ----- decoder ----- 
    decoder = conv_2d_transpose(pool4, 64, 3, strides=4, output_shape=[HEIGHT//4, WIDTH//4, 64]) #  16x downsample to 4x downsample
    decoder = conv_2d(decoder, 64, 3, activation='relu')
    pool5 = conv_2d(decoder, 64, 3, activation='relu')
 
    decoder = conv_2d_transpose(pool3, 32, 3, strides=2, output_shape=[HEIGHT//4, WIDTH//4, 32]) # 8x downsample to 4x downsample
    decoder = conv_2d(decoder, 32, 3, activation='relu')
    pool6 = conv_2d(decoder, 32, 3, activation='relu')

    pool6=merge([pool6, pool5, pool2], mode='concat', axis=3) #merge all 4x downsampled layers

    decoder = conv_2d_transpose(pool6, 16, 3, strides=4, output_shape=[HEIGHT, WIDTH, 16])
    decoder = conv_2d(decoder, 16, 3, activation='relu')
    pool6 = conv_2d(decoder, 16, 3, activation='relu')
   
    decoder = conv_2d(pool6, CHANNELS, 1)
    network = tflearn.regression(decoder, optimizer='adam', loss='mean_square') 
    
    return network

# u-net adapted network 
def build_unet(network):
    Ni=8
    #Pool1
    network_1 = conv_2d(network, Ni, 3, activation='relu') 
    network_1 = conv_2d(network_1, Ni, 3, activation='relu') 
    pool1 = max_pool_2d(network_1, 2)                             # downsampling 2x  
    #Pool2
    network_2 = conv_2d(pool1, 2*Ni, 3, activation='relu') 
    network_2 = conv_2d(network_2, 2*Ni, 3, activation='relu') 
    pool2 = max_pool_2d(network_2, 2)                            # downsampling 4x 
    #Pool3
    network_3 = conv_2d(pool2, 4*Ni, 3, activation='relu') 
    network_3 = conv_2d(network_3, 4*Ni, 3, activation='relu') 
    pool3 = max_pool_2d(network_3, 2)                            # downsampling 8x 
    #Pool4
    network_4 = conv_2d(pool3, 8*Ni, 3, activation='relu') 
    network_4 = conv_2d(network_4, 8*Ni, 3, activation='relu') 
    pool4 = max_pool_2d(network_4, 2)                            # downsampling 16x 

    #Pool5
    network_5 = conv_2d(pool4, 16*Ni, 3, activation='relu') 
    network_5 = conv_2d(network_5, 16*Ni, 3, activation='relu') 

    Unpool1 = conv_2d_transpose(network_5, 8*Ni, 3, strides=2, output_shape=[HEIGHT // 8, WIDTH // 8, 8*Ni]) 
    merge1 = merge([Unpool1, network_4], mode='concat', axis=3) # merge 
    merge1 = conv_2d(merge1, 8*Ni, 3, activation='relu')
    merge1 = conv_2d(merge1, 8*Ni, 3, activation='relu')

    Unpool2 = conv_2d_transpose(merge1, 4*Ni, 3, strides=2, output_shape=[HEIGHT // 4, WIDTH // 4, 4*Ni]) 
    merge1 = merge([Unpool2, network_3], mode='concat', axis=3) # merge 
    merge1 = conv_2d(merge1, 4*Ni, 3, activation='relu')
    merge1 = conv_2d(merge1, 4*Ni, 3, activation='relu')

    Unpool3 = conv_2d_transpose(merge1, 2*Ni, 3, strides=2, output_shape=[HEIGHT // 2, WIDTH // 2, 2*Ni]) 
    merge1 = merge([Unpool3, network_2], mode='concat', axis=3) # merge 
    merge1 = conv_2d(merge1, 2*Ni, 3, activation='relu')
    merge1 = conv_2d(merge1, 2*Ni, 3, activation='relu')
        
    Unpool4 = conv_2d_transpose(merge1, Ni, 3, strides=2, output_shape=[HEIGHT, WIDTH, Ni])
    merge1 = merge([Unpool4, network_1], mode='concat', axis=3) # merge 
    merge1 = conv_2d(merge1, Ni, 3, activation='relu')
    merge1 = conv_2d(merge1, Ni, 3, activation='relu')
   
    merge1 = conv_2d(merge1, 3, 1, activation='relu')

    network = tflearn.regression(merge1, optimizer='adam', loss='mean_square') 
    
    return network

# build tiramisu (extension of u-net)
def build_tiramisu(network):
    n=5
    Ni=8

    #Pool1
    network_1 = conv_2d(network, Ni, 3, regularizer='L2', weight_decay=0.0001) # 256
    network_1 = residual_block(network_1, n, Ni)
    network_1 = residual_block(network_1, 1, Ni)     
    pool_1 = max_pool_2d(network_1, 2)              # downsampling 2x - 128
    #Pool2
    network_2 = residual_block(pool_1, n-1, 2*Ni)           
    network_2 = residual_block(network_2, 1, 2*Ni)   
    pool_2 = max_pool_2d(network_2, 2)              # downsampling 4x - 64
    #Pool3
    network_3 = residual_block(pool_2, n-1, 4*Ni)           
    network_3 = residual_block(network_3, 1, 4*Ni)   
    pool_3 = max_pool_2d(network_3, 2)              # downsampling 8x - 32
    #Pool4
    network_4 = residual_block(pool_3, n-1, 8*Ni)           
    network_4 = residual_block(network_4, 1, 8*Ni)   
    pool_4 = max_pool_2d(network_4, 2)              # downsampling 16x - 16
    #Pool5
    network_5 = residual_block(pool_4, n-1, 16*Ni)           
    network_5 = residual_block(network_5, 1, 16*Ni)

    # /////////////////////////////////////////////////////////////////////////////////

    Unpool1 = conv_2d_transpose(network_5, 8*Ni, 3, strides=2, output_shape=[HEIGHT // 8, WIDTH // 8, 8*Ni]) 
    merge1 = merge([Unpool1, network_4], mode='concat', axis=3) # merge 
    merge1 = conv_2d(merge1, 8*Ni, 3, activation='relu')           
    merge1 = conv_2d(merge1, 8*Ni, 3, activation='relu')   #

    Unpool2 = conv_2d_transpose(merge1, 4*Ni, 3, strides=2, output_shape=[HEIGHT // 4, WIDTH // 4, 4*Ni]) 
    merge2 = merge([Unpool2, network_3], mode='concat', axis=3) # merge 
    merge2 = conv_2d(merge2, 4*Ni, 3, activation='relu')           
    merge2 = conv_2d(merge2, 4*Ni, 3, activation='relu')   

    Unpool3 = conv_2d_transpose(merge2, 2*Ni, 3, strides=2, output_shape=[HEIGHT // 2, WIDTH // 2, 2*Ni]) 
    merge3 = merge([Unpool3, network_2], mode='concat', axis=3) # merge 
    merge3 = conv_2d(merge3, 2*Ni, 3, activation='relu')
    merge3 = conv_2d(merge3, 2*Ni, 3, activation='relu')
        
    Unpool4 = conv_2d_transpose(merge3, Ni, 3, strides=2, output_shape=[HEIGHT, WIDTH, Ni])
    merge4 = merge([Unpool4, network_1], mode='concat', axis=3) # merge 
    merge4 = conv_2d(merge4, Ni, 3, activation='relu')
    merge4 = conv_2d(merge4, Ni, 3, activation='relu')
   
    final_merge = conv_2d(merge4, 3, 1, activation='relu')

    network = tflearn.regression(final_merge, optimizer='adam', loss='mean_square') 
    
    return network


# ////////////////////////////////////////////////////
#             Network identifiers
# ////////////////////////////////////////////////////
if(args.arch == "autoencoder"):
    network = build_autoencoder(network)
elif(args.arch == "fcn"):
    network = build_fcn_all(network)
elif(args.arch == "segnet_half"):
    network = build_segnet_half(network)
elif(args.arch == "unet"):
    network = build_unet(network)
elif(args.arch == "tiramisu"):
    network = build_tiramisu(network)
else:
    network = build_segnet(network)

# ////////////////////////////////////////////////////
#                Model definition
# ////////////////////////////////////////////////////
retraining = False
training_id = args.model_name

if(os.path.isfile(args.model_name + ".data-00000-of-00001")):
    retraining = True
    parts = args.model_name.split(os.sep)
    parts.reverse()
    training_id = parts[0].split('.')[0]

model = tflearn.DNN(network, checkpoint_path="./models/%s" % training_id, tensorboard_dir='./logs/', 
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95, 
                    best_checkpoint_path=None) 

# ////////////////////////////////////////////////////
#                Custom callbakcks
# ////////////////////////////////////////////////////
# callback monitor
class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, frequency=5):
        self.train_losses = []
        self.snapshot_every = frequency # saves checkpoint every 5 epochs
    
    def on_epoch_end(self, training_state, snapshot=False):
        if(self.snapshot_every is not None and training_state.epoch > 0 and 
           training_state.epoch % self.snapshot_every == 0):
            print("[INFO] Saving checkpoint model...")
            ckptname = "./models/%s-epoch%d" % (training_id, training_state.epoch)
            print("[INFO] Checkpoint: ", ckptname)
            model.save(ckptname)
            print("[INFO] Checkpoint saved!\n")

# custom callback
class LossMonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, model):
        self.lowest_loss = None
        self.model = model

    def on_epoch_end(self, training_state):
        if(self.lowest_loss is None):
            self.lowest_loss = training_state.global_loss
        elif(training_state.global_loss < self.lowest_loss):
            self.lowest_loss = training_state.global_loss

        print("[INFO] Global loss:", training_state.global_loss)

# ////////////////////////////////////////////////////
#          Training callbacks definition
# ////////////////////////////////////////////////////
saverMonitor = MonitorCallback(frequency=100)
lossMonitor  = LossMonitorCallback(model)

# training operation
if(args.data_dir is not None and args.bsize is not None):
    # training parameters
    EPOCHS = 1000 # maximum number of epochs

    print("[INFO] Batch size:", args.bsize)
    print("[INFO]     Epochs:", EPOCHS) 

    if(retraining):
        # re-training operation
        # load trained model
        print("")
        print("[INFO] Loading pre-trained model for re-training...")  
        model.load(args.model_name)
        print("[INFO] Model:", args.model_name)
        print("[INFO] Trained model loaded!\n")

    try:
        print("[INFO] Starting training operation...")
        print("[INFO] Training ID:", training_id)
        # repeats the training operation until it reaches one stop criteria
        model.fit(Xim, Xgt, n_epoch=EPOCHS, shuffle=True, show_metric=True, 
                  batch_size=args.bsize, snapshot_step=None, snapshot_epoch=False, 
                  run_id=training_id, validation_set=0, callbacks=[])

    # to stop the training at any moment by pressing Ctrl+C
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user. Pressed 'Ctrl+C'")
        pass 

    # save trained model
    if(args.freeze):
        # NOTE: use it for freezing model
        del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
        model.save("./models/%s_frozen.tflearn" % training_id)

    print("[INFO] Saving trained model...")
    
    if(retraining):
        # NOTE: use this while there is no metric to evaluate the model properly
        modelname = "./models/%s_rt.tflearn" % training_id
    else:
        modelname = "./models/%s.tflearn" % training_id

    print("[INFO] Model: ", modelname)
    model.save(modelname)
    print("[INFO] Trained model saved!\n")

else:
    # testing operation
    # load trained model
    print("[INFO] Loading pre-trained model for testing...")  
    model.load(args.model_name)
    print("[INFO] Model:", args.model_name)
    print("[INFO] Trained model loaded!\n")


# ////////////////////////////////////////////////////////////////////////////////////////////////
# testing setup
# to load images from test dir
if(args.data_dir is not None or args.test_dir is not None):
    delay = 0
    try:
        print("[INFO] Testing image folder:", args.test_dir)
        X, _ = image_dirs_to_samples(args.test_dir, resize=(WIDTH, HEIGHT), convert_gray=False, 
                                     filetypes=[".png", ".jpg", ".JPG", ".bmp"])
        
        split = len(X) // 2
        Xim = X[:split]
        OXgt = X[split:] 

        OXgt = np.array(OXgt)
        OXgt = np.reshape(OXgt, (-1, HEIGHT, WIDTH, 1))
        Xgt = []

        for index, elem in enumerate(OXgt):
            Xgt.append(cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB))
    
        Xgt = np.array(Xgt)
        Xgt = np.reshape(Xgt, (-1, HEIGHT, WIDTH, 3))
        
        print("")
        print("[INFO] Images: ", len(Xim), Xim[0].shape)
        print("[INFO] Ground: ", len(Xgt), Xgt[0].shape, "\n")
    except Exception as e:
        print("[EXCEPTION]", e)
        print("[INFO] Couldn't load test images!") 
        print("[INFO] Testing will use training data")

    nimages = len(Xim)

else:
    # load images from video capture device
    delay = 1
    nimages = -1

    try:
        args.video = int(args.video)
    except:
        pass
    
    # print video capture source
    print("[INFO] Video:", args.video, "\n")
    # initialize viceo capture variale
    cam = cv2.VideoCapture(args.video)

# image index
image_id = 0

# flag to upsample prediction image show
upsample = True

# windows definition
spacing = 50 
cv2.namedWindow("Original")
cv2.moveWindow("Original", spacing, spacing)
cv2.namedWindow("Ground Truth")
cv2.moveWindow("Ground Truth", spacing + WIDTH + 20, spacing)
cv2.namedWindow("Predicted Mask")
cv2.moveWindow("Predicted Mask", spacing, spacing + HEIGHT + spacing)
cv2.namedWindow("Predicted")
cv2.moveWindow("Predicted", spacing + WIDTH + 20, spacing + HEIGHT + spacing)

# while loop to constantly load images 
while True:
    # start measuring time
    ctime = time.time()
    
    if((args.data_dir is not None or args.test_dir is not None) and image_id < nimages):
        # load image from folder
        frame = Xim[image_id]
    elif(args.data_dir is None and args.test_dir is None and args.video is not None):
        # load image from video capture source
        ret_val, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        break
    
    # reshape test image to NHWC tensor format
    test_image = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    # NOTE: exceptional treatment when using video capture
    if(args.video is not None and args.data_dir is None and args.test_dir is None):
        test_image = test_image / 255.

    copy_test_image = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    test_image = np.reshape(test_image, (1, HEIGHT, WIDTH, CHANNELS)) 

    # output mask prediction
    prediction = model.predict(test_image)
    try:
        # for RGB outputs
        prediction = np.reshape(prediction[0], (HEIGHT, WIDTH, 3))
    except:
        # for GS outputs
        prediction = np.reshape(prediction[0], (HEIGHT, WIDTH, 1))
    
    # original image 
    original_bgr = cv2.cvtColor(copy_test_image, cv2.COLOR_RGB2BGR)
    # NOTE: exceptional treatment when using video capture
    if(args.video is not None and args.test_dir is None):
        #original_bgr = original_bgr / 255.
        pass
    
    if(args.show >= 1):
        print("\n[INFO] Show original", original_bgr.shape)
        cv2.imshow("Original", original_bgr)

    # predicted segmentation 
    try:
        prediction_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        prediction_bgr = np.absolute(prediction)
    except Exception as convert_pred_colorspace:
        print("[EXCEPTION]", convert_pred_colorspace)
        print("[EXCEPTION] Prediction can't be converted to BGR colorspace!")
        prediction_bgr = prediction.copy()
        pass
    
    if(args.show >= 1):
        print("[INFO] Show predicted mask", prediction_bgr.shape)
        cv2.imshow("Predicted Mask", prediction)

    # ground truth
    if(args.data_dir is not None or args.show >= 0):
        try:
            gtruth = cv2.cvtColor(Xgt[image_id], cv2.COLOR_RGB2BGR)
        except Exception as convert_gt_colorspace:
            print("[EXCEPTION]", convert_gt_colorspace)
            print("[EXCEPTION] Ground truth can't be converted to BGR colorspace!")
            gtruth = Xgt[image_id]
        annotations = 0.5 * original_bgr + 0.5 * gtruth
        
        print("[INFO] Show ground truth")
        cv2.imshow("Ground Truth", annotations)

    # prediction overlay 
    if(args.show >= 0):
        overlay = 0.5 * original_bgr + 0.5 * prediction_bgr
        if(upsample):
            overlay = cv2.resize(overlay, (WIDTH*2, HEIGHT*2), interpolation=cv2.INTER_CUBIC)
        #overlay = overlay * 255

        print("[INFO] Show predicted")
        cv2.imshow("Predicted", overlay)

    if(args.save):
        cv2.imwrite("./out/seg-%d.png" % image_id, overlay)
    
    ctime = time.time() - ctime

    print("[INFO] Image %d of %d | Time %.3f" % (image_id+1, nimages, ctime), end='\n\n')
    
    image_id += 1
    
    key = cv2.waitKey(delay)
    if(key == 27):
        # pressed ESC
        print("[INFO] Pressed ESC")
        break

print("\n[INFO] All done!\a")