"""
Testing script for trained models
Tensorflow and TFLearn
"""

from __future__ import division, print_function, absolute_import

import sys
import os
import platform
import time
import cv2
import glob
import shutil
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflearn
import numpy as np

from tflearn.layers.core import input_data
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from utils import architectures, dataset, classifier, helper
from colorama import init
from termcolor import colored
from matplotlib import pyplot as plt

# init colored print
init()

true_cases = ['true', 't', 'yes', '1']

# argument parser
custom_formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=2000)
parser = argparse.ArgumentParser(description="Testing script for CNNs.",
                                 formatter_class=custom_formatter_class,
                                 prefix_chars='-') 
# required arguments
parser.add_argument("--data_dir", required=True, help="<REQUIRED> directory to the training data", type=str)
parser.add_argument("--arch", required=True, help="<REQUIRED> architecture name", type=str)
parser.add_argument("--model", required=True, help="<REQUIRED> run identifier (id) / model's path", type=str)
# optional arguments
parser.add_argument("--test_dir", required=False, help="path to test images", type=str, default=None)
parser.add_argument("--height", required=False, help="images height (default=64)", default=64, type=int)
parser.add_argument("--width", required=False, help="images width (default=64)", default=64, type=int)
parser.add_argument("--val_set", required=False, help="percentage of training data to validation (default=1)", default=1, type=float)
parser.add_argument("--gray", required=False, help="convert images to grayscale (default=False)", default=False, type=lambda s: s.lower() in true_cases)
parser.add_argument("--pproc", required=False, help="enable/disable pre-processing (default=True)", default=True, type=lambda s: s.lower() in true_cases)
parser.add_argument("--aug", required=False, nargs="+", help="enable data augmentation (default=[])", default=[])
parser.add_argument("--eval_crit", required=False, help="classification confidence threshold (default=0.75)", default=0.75, type=float)
# TODO
parser.add_argument("--video", required=False, help="use video capture device/video (default=0)", default=0)
parser.add_argument("--save", required=False, help="save output image (default=False)", default=False, type=lambda s: s.lower() in true_cases)

# parse arguments
args = parser.parse_args()
# print args
print(args,"\n")

# images properties
HEIGHT = args.height
WIDTH  = args.width

""" """
# load dataset and get image dimensions
try:
    if(args.val_set):
        CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, _, _ = dataset.load_dataset_windows(args.data_dir, HEIGHT, WIDTH, shuffled=True, 
                                                                                            validation=args.val_set, mean=False, gray=args.gray, 
                                                                                            save_dd=False, data_aug=args.aug)
        classifier.HEIGHT   = HEIGHT
        classifier.WIDTH    = WIDTH
        classifier.IMAGE    = HEIGHT
        classifier.CHANNELS = CHANNELS
    else:
        CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, _, _, _, _= dataset.load_dataset_windows(args.data_dir, HEIGHT, WIDTH, shuffled=True, 
                                                                                         save_dd=False, data_aug=args.aug)
        Xv = Yv = []

except Exception as e:
    print("[EXCEPTION]", e)

    X = Y = Xv = Yv = None
    # /////////////
    # set manually 
    CHANNELS = 3
    CLASSES  = 2
    # /////////////
    print("[INFO] Loading files from index file")
    print("[INFO] Path:", args.data_dir)
    pass

# load test images
Xt = Yt = None
if(args.test_dir is not None):
    _, Xt, Yt, _, _, _, _, _, _, _ = dataset.load_dataset_windows(args.test_dir, HEIGHT, WIDTH, shuffled=True, mean=False, 
                                                                  gray=args.gray, save_dd=True, data_aug=args.aug)


# to load CIFAR-10 or MNIST dataset
if(False):
    print("Loading dataset (from directory)...")

    CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, Xt, Yt = dataset.load_mnist_dataset(data_dir=args.data_dir)
    #CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, Xt, Yt = dataset.load_cifar10_dataset(data_dir=args.data_dir)

    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS

    print("\t         Path:",args.data_dir)
    print("\tShape (train):",X.shape,Y.shape)
    print("\tShape   (val):",Xv.shape,Yv.shape)
    print("Data loaded!\n")


# Real-time data preprocessing (samplewise or featurewise)
img_prep = None
if(args.pproc):
    img_prep = ImagePreprocessing()
    img_prep.add_samplewise_zero_center()
    img_prep.add_samplewise_stdnorm()      
    #img_prep.add_zca_whitening()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=5.)

# computational resources definition (made changes on TFLearn's config.py)
tflearn.init_graph(num_cores=8, allow_growth=True)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS],    # shape=[None,IMAGE, IMAGE] for RNN
                     data_preprocessing=img_prep,              # NOTE: always check PP
                     data_augmentation=None)                   # NOTE: always check DA

network,_ = architectures.build_network(args.arch, network, CLASSES, None)

# model definition
model = tflearn.DNN(network)  

print("[INFO] Eval crit.:", args.eval_crit)
print("[INFO] Validation:", args.val_set*100 , "%\n")

# load model to figure out if there is something wrong 
print("[INFO] Loading trained model...")  
model.load(args.model)
print("[INFO] Model:", args.model)
print("[INFO] Trained model loaded!\n")    

# final evaluation with the best model
if(Xv is not None):
    stime = time.time()
    #train_acc, _ = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=args.eval_crit)
    val_acc, val_acc_nc  = classifier.my_evaluate(model, Xv, Yv, batch_size=128, criteria=args.eval_crit)
    ftime = time.time() - stime
else:
    val_acc  = 0
    val_acc_nc = 0
    ftime = 0

# write accuracy's values on screen
helper.print_accuracy(name="Final Eval", train_acc=None, val_acc=val_acc, test_acc=val_acc_nc, 
                      min_acc=None, time=None, ctime=ftime, color="green")

# shows image and predicted class
# NOTE: Turn to false when scheduling many trainings
print(colored("[INFO] Showing dataset performance", "yellow"))

# NOTE: Choose an image set
if(Xv is not None):
    image_set = Xv
    label_set = Yv
else:
    image_set = glob.glob(args.data_dir + "*")

len_is = len(image_set) # lenght of test set
if(len_is < 1):
    sys.exit(colored("[INFO] Test set has no images!","yellow"))

bp = 0                # badly predicted counter
wp = 0                # well predicted counter  
separate = False      # separates images with help of a trained model
show_image = True     # flag to (not) show tested images
cmatrix = None        # NOTE: manually set by user

if(args.save):
    gridx = 5
    gridy = 1
    fig, axes = plt.subplots(gridy, gridx)
    axes = axes.flat
    keep_save = True

if(cmatrix is not None):
    fcsv = open(cmatrix + "_test_cmatrix.txt","w+")
    fcsv.write("predicted,label\n")

j=0
for i in np.arange(0,len_is):
    # classify the image
    if(Xv is None):
        image = cv2.imread(image_set[i], 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (HEIGHT, WIDTH))
        image = np.reshape(image, (-1,HEIGHT,WIDTH,CHANNELS))
        probs = model.predict(image)
    else:
        probs = model.predict(image_set[np.newaxis, i])
    probs = np.asarray(probs)
    
    # sorted indexes by confidences 
    predictions = np.argsort(-probs,axis=1)[0]
    guesses = predictions[0:2]
    
    # get best classification confidence
    ci = int(guesses[0])
    confidence = probs[0][ci]
    if(Yv is not None):
        true_label = np.argmax(label_set[i])
    
    # write on test confusion matrix
    if(cmatrix is not None):
        fcsv = open(cmatrix + "_test_cmatrix.txt","a+")
        fcsv.write("%d,%d\n" % (guesses[0],true_label))
    
    # resize the image to 128 x 128
    #image = image_set[i]
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image_set[i], (HEIGHT, WIDTH))

    # ///////////////// to separate images by its classes ///////////////// 
    if(separate):
        print("Predicted: {0}, Confidence: {1:3.2f}, Second guess: {2} ({3}/{4})".format(guesses[0],confidence,guesses[1],i+1,len_is))
        # NOTE: Expecting something like "./samples\\0.jpg"
        current_image = image_set[i]
        src = current_image

        dest_file = current_image.split("\\")
        dest_file.reverse()
        dest_file = dest_file[0]

        dest_folder = "./dataset/numbers/augmented_v7/" # NOTE: set manually by user
        dest_folder = "E:/Testes/Bruno/BrunoTestesDeepLearning/ImagensEboosterParaTestes/Classificado/"

        # NOTE: move images by confirming them manually
        if(False):
            cv2.imshow("Test image", image)
            key = cv2.waitKey(0)
        
            if(key == 13):
                # enter
                # NOTE: Always adapt the destination folder
                dest = dest_folder + str(guesses[0]) + "/_" + dest_file
                shutil.copy(src,dest)
            elif(key > 47 and key < 58):
                # works for [0,9] but not for >9
                dest = dest_folder + str(key-48) + "/_" + dest_file
                shutil.copy(src,dest)
                pass
            elif(key == 27):
                # escape
                break
            else:
                # any other key to skip to the next image
                pass
        else:
            # NOTE: move files automatically
            if(confidence >= 0.75):
                dest = dest_folder + str(guesses[0]) + "/" + dest_file
            else:
                dest = dest_folder + "2/" + dest_file
            
            shutil.copy(src,dest)
    
    else:
        # ///////////////// shows badly predicted images /////////////////
        if(guesses[0] != true_label):
            bp += 1
            # shows images if flag is true 
            if(show_image):
                print("Predicted: {0:2d}, Actual: {1:2d}, Confidence: {2:3.3f}, Second guess: {3:2d}".format(int(guesses[0]), np.argmax(label_set[i]), confidence, int(guesses[1])))
                rgb = np.fliplr(image.reshape(-1,CHANNELS)).reshape(image.shape)
                factor = 1
                rgb = cv2.resize(rgb, (WIDTH*factor,HEIGHT*factor), interpolation=cv2.INTER_CUBIC)
                # add predicted label to the image
                cv2.putText(rgb, str(true_label), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
                cv2.putText(rgb, str(guesses[0]), (5, HEIGHT*factor-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
                cv2.putText(rgb, str(guesses[1]), (WIDTH*factor-20, HEIGHT*factor-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
                # shows image
                cv2.imshow("Test image", rgb)
                key = cv2.waitKey(0)

                if(args.save and keep_save):
                    if(j==0):
                        final_frame = rgb.copy()
                    else:
                        try:
                            final_frame = cv2.hconcat((final_frame, rgb))
                            axes[j].imshow(rgb, cmap='gray')
                            axes[j].set_xticks([])
                            axes[j].set_yticks([])
                            j += 1
                        except:
                            keep_save = False

            if(key == 27):
                # pressed Esc
                cv2.destroyWindow("Test image") 
                show_image = False
        else:
            if(confidence > args.eval_crit):
                wp += 1

if(args.save):
    cv2.imshow("final", final_frame)
    cv2.imwrite("fframe.jpg", final_frame)
    plt.show()

if(cmatrix is not None):
    fcsv.close()

print(colored("[INFO] %d badly predicted images in a total of %d (Error rate %.4f)" % (bp,len_is,bp/len_is),"yellow"))
print(colored("[INFO] %d well predicted images (confidence > %.2f) in a total of %d (Acc. %.4f)" % (wp,args.eval_crit,len_is,wp/len_is),"yellow"))

# sound an ending beep
print(colored("[INFO] Testing complete!\a","green"))