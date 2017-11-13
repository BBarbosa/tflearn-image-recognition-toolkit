from __future__ import division, print_function, absolute_import
import sys, os, platform, time, cv2, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data,dropout,fully_connected,flatten
from tflearn.layers.conv import conv_2d,max_pool_2d,highway_conv_2d,avg_pool_2d,upsample_2d,upscore_layer,conv_2d_transpose
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization,batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.data_utils import image_dirs_to_samples
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import winsound as ws
import numpy as np
from colorama import init
from termcolor import colored
from PIL import Image

# NOTE: change if you want a specific size

#HEIGHT = 256
#HEIGHT = 240

HEIGHT = 192
WIDTH  = 608

HEIGHT = 96 
WIDTH  = 320

#HEIGHT = 160 
#WIDTH  = 320

CHANNELS = 3

# init colored print
init()

# argument parser
parser = argparse.ArgumentParser(description="Automatic image segmentation using Deep Learning.",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("--train_dir",required=False,help="directory to the training data",type=str)
parser.add_argument("--architecture",required=True,help="architecture name",type=str)
parser.add_argument("--batch_size",required=False,help="training batch size",type=int)
parser.add_argument("--run_id",required=True,help="run identifier (id) / model's path",type=str)
# optional arguments
parser.add_argument("--test_dir",help="path to test images",type=str)
parser.add_argument("--video",help="use video capture device/video")
parser.add_argument("--save",help="save output image (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--freeze",help="flag to freeze model (boolean)",default=False,type=lambda s: s.lower() in ['true', 't', 'yes', '1'])

# parse arguments
args = parser.parse_args()

# clears screen and shows OS
OS = platform.system() 

if(OS == 'Windows'):
    os.system('cls')
else:
    os.system('clear')
print("Operating System: %s\n" % OS)

# print args
print(args,"\n")

# load dataset and get image dimensions
if(args.train_dir is not None and args.batch_size is not None):
    X,_ = image_dirs_to_samples(args.train_dir, resize=(WIDTH,HEIGHT), convert_gray=False, 
                                filetypes=[".png",".jpg",".bmp"])
    
    #Xgt,_ = image_dirs_to_samples(args.train_dir, resize=(WIDTH,HEIGHT), convert_gray=False, 
    #                            filetypes=[".png",".jpg",".bmp"])

    # images / ground truth split
    split = len(X) // 2
    Xim = X[:split]
    Xgt = X[split:] 

    print("")
    print("Images: ", len(Xim), Xim[0].shape)
    print("Ground: ", len(Xgt), Xgt[0].shape, "\n")
else:
    print("[INFO] Training images not set\n")

# Real-time data preprocessing  
img_prep = ImagePreprocessing()
img_prep.add_samplewise_zero_center()   # per sample (featurewise is a global value)
img_prep.add_samplewise_stdnorm()       # per sample (featurewise is a global value)

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=10.)

# computational resources definition (made changes on TFLearn's config.py)
tflearn.init_graph(num_cores=8,allow_growth=True)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS],    # shape=[None,IMAGE, IMAGE] for RNN
                     data_preprocessing=None,                  # NOTE: always check PP
                     data_augmentation=None)                   # NOTE: always check DA

def my_loss(y_pred, y_true):
    return tflearn.objectives.weak_cross_entropy_2d(y_pred, y_true, num_classes=4)

def my_metric(y_pred,t_true):
    return tflearn.metrics.top_k_op(y_pred,t_true,k=3)

# ---
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
    decoder = conv_2d_transpose(encoder, 64, 3, strides=2, output_shape=[HEIGHT//4,WIDTH//4,64])
    decoder = conv_2d(decoder, 64, 3, activation='relu')
    decoder = conv_2d(decoder, 64, 3, activation='relu')
    
    decoder = conv_2d_transpose(decoder, 32, 5, strides=2, output_shape=[HEIGHT//2,WIDTH//2,32])
    decoder = conv_2d(decoder, 32, 5, activation='relu')
    decoder = conv_2d(decoder, 32, 5, activation='relu')

    decoder = conv_2d_transpose(decoder, 16, 7, strides=2, output_shape=[HEIGHT,WIDTH,16])
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
# fully-convolutional network
def build_fcn_all(network):
    #Pool1
    conv1 = conv_2d(network, 8, 7, activation='relu')
    pool1 = max_pool_2d(conv1,2)
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
    #network_32_UP4 = upsample_2d(network_32,4)
    #network_16_UP2  = upsample_2d(network_16,2)
    #network_3_UP8   = upsample_2d(network_3,8)
    pool4_x2 = upsample_2d(pool4,2)
    conv7_x4 = upsample_2d(conv7,4)
    #network_8 = merge([network_32_UP4, network_4_UP2, network_3], mode='concat', axis=3)
    fcn_8s = merge([pool3, pool4_x2, conv7_x4], mode='concat', axis=3)
    fcn_8s = conv_2d(fcn_8s, 3, 1, activation='relu')
    ##end FCN-8s -----------------------------------
    
    out = conv_2d(fcn_8s, 3, 1, activation='relu')
    out = upsample_2d(out,8)
    #network_8 = upscore_layer(network_8,num_classes=3,kernel_size=2,strides=8,shape=[384,1216,3])
    
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
    pool1 = max_pool_2d(network_1,2)
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
   
    decoder = conv_2d(pool6, 3, 1)
    network = tflearn.regression(decoder, optimizer='adam', loss='mean_square') 
    
    return network

# segnet-like half filters
def build_segnet_half(network):
    #Pool1
    network_1 = conv_2d(network, 8, 3, activation='relu') #output 2x_downsampled
    network_1 = conv_2d(network_1, 8, 3, activation='relu') #output 2x_downsampled
    pool1 = max_pool_2d(network_1,2)
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
   
    decoder = conv_2d(pool6, 3, 1)
    network = tflearn.regression(decoder, optimizer='adam', loss='mean_square') 
    
    return network
# ---

if(args.architecture == "autoencoder"):
    network = build_autoencoder(network)
elif(args.architecture == "fcn"):
    network = build_fcn_all(network)
elif(args.architecture == "segnet_half"):
    network = build_segnet_half(network)
else:
    network = build_segnet(network)

# model definition
model = tflearn.DNN(network, checkpoint_path="models/%s" % args.run_id, tensorboard_dir='logs/',
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95,
                    best_checkpoint_path=None)  

# callback monitor
class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self,frequency=None):
        self.train_losses = []
        self.snapshot_every = frequency # saves checkpoint every 5 epochs
        pass
    
    def on_epoch_end(self, training_state, snapshot=False):
        if(self.snapshot_every is not None and training_state.epoch > 0 and training_state.epoch % self.snapshot_every == 0):
            print("[INFO] Saving checkpoint model...")
            ckptname = "models/%s-epoch%d" % (args.run_id,training_state.epoch)
            print("[INFO] Checkpoint: ",ckptname)
            model.save(ckptname)
            print("[INFO] Checkpoint saved!\n")

saverMonitor = MonitorCallback(frequency=100)

# training operation
if(args.train_dir is not None):
    # training parameters
    EPOCHS = 1000 # maximum number of epochs

    print("[INFO] Batch size:", args.batch_size)
    print("[INFO] Epochs:", EPOCHS) 

    try:
        # repeats the training operation until it reaches one stop criteria
        model.fit(Xim, Xgt, n_epoch=EPOCHS, shuffle=True, show_metric=True, 
                  batch_size=args.batch_size, snapshot_step=None, snapshot_epoch=False, 
                  run_id=args.run_id, validation_set=0.1, callbacks=saverMonitor)

    # to stop the training at any moment by pressing Ctrl+C
    except KeyboardInterrupt:
        pass 

    # save trained model
    if(args.freeze):
        # NOTE: use it for freezing model
        del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]

    print("[INFO] Saving trained model...")
    modelname = "models/%s.tflearn" % args.run_id
    print("[INFO] Model: ",modelname)
    model.save(modelname)
    print("[INFO] Trained model saved!\n")

else:
    print("[INFO] Loading trained model...")  
    model.load(args.run_id)
    print("[INFO] Model: ",args.run_id)
    print("[INFO] Trained model loaded!\n")

    if(args.test_dir):
        try:
            Xim,_ = image_dirs_to_samples(args.test_dir, resize=(WIDTH,HEIGHT), convert_gray=False, 
                                          filetypes=[".png",".jpg",".JPG",".bmp"]) 
            print("")
            print("Images: ", len(Xim), Xim[0].shape, "\n")
        except:
            print("[INFO] Couldn't load test images!")
            pass  

##############################
#----------- CAMERA ----------
#if(args.video):
#    delay = 1
#    nimages = -1
#
#i=0
#try:
#    args.video = int(args.video)
#except:
#    pass
#
#print("[INFO] Video:",args.video,"\n")
#
#cam = cv2.VideoCapture(args.video)
#while(not cam.isOpened()):
#    cam = cv2.VideoCapture(args.video)
#
#while True:
#    ret_val, cam_image = cam.read()
#    while(not ret_val):
#        ret_val, cam_image = cam.read()
#    
#    stime = time.time()
#
#    cam_image = cv2.cvtColor(cam_image,cv2.COLOR_BGR2RGB)
#    test_image = cv2.resize(cam_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)
#    test_image2 = cv2.resize(cam_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)
#
#    test_image = test_image / 255.
#    #test_image2 = test_image2 / 255.
#
################################


################################
#-------- IMAGES FOLDER --------
nimages = len(Xim)
delay = 0
for i in range(nimages):
    stime = time.time()
################################ 

    test_image = np.reshape(Xim[i],(1, HEIGHT, WIDTH, CHANNELS)) 
    #test_image = np.reshape(test_image,(1, HEIGHT, WIDTH, 3)) # video capture
    
    predicts = model.predict(test_image)
    pred_image = np.reshape(predicts[0], (HEIGHT, WIDTH, CHANNELS))
    
    ### original image --------------------------------------------
    original = cv2.cvtColor(Xim[i],cv2.COLOR_RGB2BGR)
    
    #original = test_image2                                 # video capture
    #original = cv2.cvtColor(test_image2,cv2.COLOR_RGB2BGR) # video capture
    #original = original / 255.                             # video capture
    
    #cv2.imshow("Original",original)

    ### predicted segmentation ------------------------------------
    pred_image = np.absolute(pred_image)
    predicted = cv2.cvtColor(pred_image,cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Predicted Mask",predicted)

    ### ground truth
    if(args.train_dir is not None):
        gtruth = cv2.cvtColor(Xgt[i],cv2.COLOR_RGB2BGR)
        #cv2.imshow("Ground truth",gtruth)

        annotations = 0.5 * original + 0.5 * gtruth
        cv2.imshow("Ground Truth",annotations)

    ### prediction overlay 
    overlay = 0.5 * original + 0.5 * predicted
    cv2.imshow("Predicted",overlay)

    if(args.save):
        cv2.imwrite(".\\out\\%d.png" % i,overlay)
    
    ftime = time.time() - stime

    print("\rImage %d of %d | Time %.3f" % (i+1,nimages,ftime), end='')
    
    if(args.video is not None):
        i += 1
    
    key = cv2.waitKey(delay)
    if(key == 27):
        # pressed ESC
        print("")
        break

print("\n[INFO] All done!")