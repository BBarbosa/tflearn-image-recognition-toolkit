from __future__ import division, print_function, absolute_import

import os,sys,time,platform,six
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflearn
import winsound as ws  # for windows only
import tensorflow as tf
import scipy.ndimage

from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data
from tflearn.data_augmentation import ImageAugmentation
from utils import architectures

import numpy as np 
from PIL import Image,ImageDraw

from matplotlib import pyplot as plt
from colorama import init
from termcolor import colored

# init colored print
init()

IMAGE  = 128
HEIGHT = 128
WIDTH  = 128

# control flags for extra features
saveOutputImage = False

# return a color according to the class
def getColor(x):
    return {
        0 : (255,0,0),      # red
        1 : (0,255,0),      # green
        2 : (0,0,255),      # blue
        3 : (255,255,0),    # yellow
        4 : (255,128,5),    # orange  
        5 : (255,20,147),   # pink
        6 : (0,255,255),    # cyan
        7 : (255,255,255),  # white
        8 : (128,0,128),    # purple
    }[x]

# function that clears the screen and shows the current OS
def clear_screen():
    # specify OS
    OS = platform.system() 
    
    # clear screen and show OS
    if(OS == 'Windows'):
        os.system('cls')
    else:
        os.system('clear')
    print("Operating System: %s\n" % OS)

# function that classifies a single image and returns labelID and confidence
def classify_single_image(model,image,label=None):
    probs = model.predict(image)
    index = np.argmax(probs)
    prob  = probs[index] 
    
    if(label):
        print("    Label: ",label)
        print("Predicted: ",index)

    return index,prob

# function that classifies a image by a sliding window (in extreme, 1 window only)
def classify_sliding_window(model,image_list,label_list,runid,nclasses):
    if(len(image_list) != len(label_list)):
        sys.exit()
        sys.exit(colored("ERROR: Image and labels list must have the same lenght!","red"))
    
    accuracies = []
    cmatrix    = []

    # Load the image file (need pre-processment)
    for image,classid in zip(image_list,label_list):
        hDIM,wDIM  = image.size     
        img        = np.asarray(image)
        img        = scipy.misc.imresize(img, (hDIM,wDIM), interp="bicubic").astype(np.float32, casting='unsafe')
        img       -= scipy.ndimage.measurements.mean(img)           # confirmed. check data_utils.py on github
        img       /= np.std(img)                                    # confirmed. check data_utils.py on github

        BLOCK     = 8                                               # side of square block for painting: BLOCKxBLOCK. Assume BLOCK <= IMAGE  
        padding   = (IMAGE - BLOCK) // 2                            # padding for centering sliding window
        nhDIM     = hDIM - 2*padding                                                
        nwDIM     = wDIM - 2*padding                                
        hshifts   = nhDIM // BLOCK                                  # number of sliding window shifts on height
        wshifts   = nwDIM // BLOCK                                  # number of sliding window shifts on width
        total = hshifts*wshifts                                     # total number of windows
        counts = [0] * nclasses                                     # will count the occurences of each class. resets at every image

        if(saveOutputImage):
            background = image
            segmented = Image.new('RGB', (wDIM,hDIM), "black")      # create mask for segmentation

        # sliding window (center)
        print("Classification started...")
        print("\t   Size: ", wDIM, "x", hDIM)
        print("\t  Block: ", BLOCK)

        # start measuring time
        start_time = time.time()
        for i in range(0,hshifts):
            h = i*BLOCK
            for j in range(0,wshifts):
                w = j*BLOCK
                img2 = img[h:h+HEIGHT,w:w+WIDTH]
                img2 = np.reshape(img2,(1,HEIGHT,WIDTH,3))
                #img2 = np.reshape(img2,(1,IMAGE,IMAGE))    # for RNN 

                probs = model.predict(img2)
                index = np.argmax(probs)
                counts[index] = counts[index] + 1
                val = probs[0][index]
                color = getColor(index)

                # segment block
                if(saveOutputImage):
                    for k in range(h+padding,h+BLOCK+padding):
                            for z in range(w+padding,w+BLOCK+padding):
                                segmented.putpixel((z,k),color)
            
        # stop measuring time
        cls_time = time.time() - start_time
        print("\t   Time:  %s seconds" % cls_time)
        print("Classification done!\n")
        
        # save output image options
        if(saveOutputImage):
            background = background.convert("RGBA")
            segmented = segmented.convert("RGBA")
            new_img = Image.blend(background, segmented, 0.3)
            
            output  = "epoch_%d_C%d.png" % (runid,classid)
            new_img.save(output,"PNG")        

        # Error check if not a collage
        if (classid != -1):
            acc   = counts[classid] / total * 100   # accuracy
            most  = np.argmax(counts)               # most probable class
            print("Error check...")
            print("\t  Class: ", classid)
            print("\t  Total: ", total)
            print("\t Counts: ", counts)
            print("\tPredict: ", most)
            print("\t    Acc: ", acc, "%")
            print("Error checked!\n")

            error_file = "epoch_%d_error.txt" % runid
            ferror = open(error_file, "a+")
                 
            array = ','.join(str(x) for x in counts)   # convert array of count into one single string
            ferror.write("Total: %5d | Class: %d | [%s] | Acc: %.2f | Time: %f\n" % (total,classid,array,acc,cls_time))
            ferror.close()

            accuracies.append(acc)

    avg_acc = sum(accuracies) / len(image_list)

    return accuracies,avg_acc,max(accuracies),min(accuracies)
