from __future__ import division, print_function, absolute_import
import sys, os, platform, time, cv2, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data
from tflearn.data_utils import image_dirs_to_samples
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import winsound as ws
import numpy as np
from utils import architectures,classifier
from colorama import init
from termcolor import colored
from PIL import Image

# NOTE: change if you want a specific size
HEIGHT = 192 
WIDTH  = 608
CHANNELS = 3
CLASSES = 2

# init colored print
init()

# clears screen and shows OS
classifier.clear_screen()

# argument parser
parser = argparse.ArgumentParser(description="Automatic image segmentation using Deep Learning.",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("train_dir",help="directory to the training data",type=str)
parser.add_argument("architecture",help="architecture name",type=str)
parser.add_argument("batch_size",help="training batch size",type=int)
parser.add_argument("run_id",help="run identifier (id)",type=str)
# optional arguments
parser.add_argument("--test_dir",help="path to test images",type=str)
parser.add_argument("--camera",help="use video capture device (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--save",help="save output image (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])

# parse arguments
args = parser.parse_args()
print(args,"\n")

# load dataset and get image dimensions
X,_ = image_dirs_to_samples(args.train_dir, resize=(WIDTH,HEIGHT), convert_gray=False, filetypes=".png")

# images / ground truth split
split = len(X) // 2
Xim = X[:split]
Xgt = X[split:] 

print("")
print("Images: ", len(Xim), Xim[0].shape)
print("Ground: ", len(Xgt), Xgt[0].shape, "\n")

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
tflearn.init_graph(num_cores=4,gpu_memory_fraction=0.4,allow_growth=True)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS],    # shape=[None,IMAGE, IMAGE] for RNN
                     data_preprocessing=None,                  # NOTE: always check PP
                     data_augmentation=None)                   # NOTE: always check DA

network,_ = architectures.build_network(args.architecture,network,CLASSES)

# model definition
model = tflearn.DNN(network, checkpoint_path="models/%s" % args.run_id, tensorboard_dir='logs/',
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95,
                    best_checkpoint_path=None)  

# training parameters
EPOCHS = 1000                      # maximum number of epochs 
print("Batch size:", args.batch_size)
print("    Epochs:", EPOCHS, "\n")
use_train_data = True

# training operation
if(not args.test_dir):
    try:
        # repeats the training operation until it reaches one stop criteria
        model.fit(Xim, Xgt, n_epoch=EPOCHS, shuffle=True, show_metric=True, 
                  batch_size=args.batch_size, snapshot_step=False, snapshot_epoch=False, 
                  run_id=args.run_id, validation_set=None, callbacks=None)

    # to stop the training at any moment by pressing Ctrl+C
    except KeyboardInterrupt:
        pass 

    # save trained model
    print("Saving trained model...")
    modelname = "models/%s.tflearn" % args.run_id
    print("\tModel: ",modelname)
    model.save(modelname)
    print("Trained model saved!\n")

else:
    print("Loading trained model...")  
    model.load("models/%s.tflearn" % args.run_id)
    print("\tModel: ","models/%s.tflearn" % args.run_id)
    print("Trained model loaded!\n")

    try:
        Xim,_ = image_dirs_to_samples(args.test_dir, resize=(WIDTH,HEIGHT), convert_gray=False, filetypes=".png") 
        print("")
        print("Images: ", len(Xim), Xim[0].shape, "\n")
        use_train_data = False
    except:
        print(colored("INFO: Couldn't load test images!","yellow"))
        
        pass  

# show results
nimages = len(Xim)
delay = 0

if(args.camera):
    delay = 150
    nimages = -1

# ---------- CAMERA ----------
i=0
cam = cv2.VideoCapture(0)
while(not cam.isOpened()):
    cam = cv2.VideoCapture(0)

while True:
    ret_val, cam_image = cam.read()
    while(not ret_val):
        ret_val, cam_image = cam.read()
    
    cam_image = cv2.cvtColor(cam_image,cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(cam_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)
    test_image2 = cv2.resize(cam_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)

#for i in range(nimages):
    stime = time.time()
    #test_image = np.reshape(Xim[i],(1, HEIGHT, WIDTH, 3)) 
    test_image = np.reshape(test_image,(1, HEIGHT, WIDTH, 3)) 
    predicts = model.predict(test_image)
    pred_image = np.reshape(predicts[0], (HEIGHT, WIDTH, 3))
    ftime = time.time() - stime
    
    # original image
    original = test_image2
    #original = cv2.cvtColor(Xim[i],cv2.COLOR_RGB2BGR)
    original = cv2.cvtColor(test_image2,cv2.COLOR_RGB2BGR)
    #cv2.imshow("Original",original)

    # predicted segmentation
    predicted = np.array(pred_image).astype(np.float32, casting='unsafe')
    predicted = cv2.cvtColor(predicted,cv2.COLOR_RGB2BGR)
    #cv2.imshow("Predicted Mask",predicted)

    # ground truth
    if(use_train_data and not args.camera):
        gtruth = cv2.cvtColor(Xgt[i],cv2.COLOR_RGB2BGR)
        #cv2.imshow("Ground truth",gtruth)

        annotations = 0.5 * original + 0.5 * gtruth
        cv2.imshow("Ground Truth",annotations)

    # prediction overlay 
    overlay = 0.5 * original + 0.5 * predicted
    cv2.imshow("Predicted",overlay)

    if(args.save):
        cv2.imwrite(".\\out\\%d.png" % i,overlay)
    
    print("\rImage %d of %d | Time %.3f" % (i,nimages,ftime), end='')
    i += 1
    
    key = cv2.waitKey(delay)
    if(key == 27):
        # pressed ESC
        print("")
        sys.exit(0)