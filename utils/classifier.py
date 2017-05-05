from __future__ import division, print_function, absolute_import

import os,sys,time,platform,six,socket
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

IMAGE  = 32
HEIGHT = 32
WIDTH  = 32

minimum = min(IMAGE, HEIGHT, WIDTH)

# control flags for extra features
saveOutputImage = False
showProgress    = False and saveOutputImage

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
    """
    Function that classifies one single image. If is passed a label to confirm,
    it shows the prediction and its confidence. Assuming that image have the 
    same dimensions as training images.

    Params:
        `model` - trained model
        `image` - image to be classified
        `label` (optional) - label of the corresponding image

    Return: Image's labelID and confidence
    """

    image = np.reshape(image,(1,HEIGHT,WIDTH,3))
    ctime = time.time()
    probs = model.predict(image)
    ctime = time.time() - ctime
    index = np.argmax(probs)
    prob  = probs[0][index] 
    
    if(isinstance(label,int)):
        print("    Label:",label)
    
    print("Predicted: %d (%.2f)\n" % (index,prob))
    print("Time: %.3f seconds" % ctime)
    
    return index,prob

# function that classifies a set of images and returns their labelIDs and confidence
def classify_set_of_images(model,images_list,labels_list=None):
    """
    Function that classifies a set of images. If is passed a label list to confirm,
    it shows the prediction and its confidence. Assuming that images have the 
    same dimensions as training images.

    Params:
        `model` - trained model
        `images_list` - set of images to be classified
        `labels_list` (optional) - set of labels of the corresponding images

    Return: Images' labelID and confidence
    """

    length      = len(images_list)
    indexes     = [-1] * length
    confidences = [-1] * length

    if(labels_list):
        if(len(labels_list) != length):
            sys.exit("ERROR! Images and labels lists must have the same lenght!")

    for image in images_list:
        image = np.reshape(image,(1,HEIGHT,WIDTH,3))

    ctime = time.time()
    probs = model.predict(images_list)
    ctime = time.time() - ctime
    
    for i,vals in enumerate(probs):
        indexes[i]     = np.argmax(vals)
        confidences[i] = vals[indexes[i]] 
    
    if(labels_list):
        print("Label | Predict | Confidence")
        print("----------------------------")
        for i in range(0,length):
            print("  %2d  |   %2d    |    %.2f" % (labels_list[i],indexes[i],confidences[i]))
    
    print("\nTime: %.3f seconds" % ctime)

    return indexes,confidences

# function that classifies a image by a sliding window (in extreme, 1 window only)
def classify_sliding_window(model,image_list,label_list,runid,nclasses):
    """
    Function that classifies a set of images through a sliding window. In an extreme
    situation, it classifies just only one window.

    Params:
        `model` - trained model
        `images_list` - list of images to be classified (already loaded)
        `labels_list` - list of labels of the corresponding images (use [-1] for collages)
        `runid` - ID for this classification
        `nclasses` - number of classes

    Return: Images' labelID and confidence
    """
    
    if(len(image_list) != len(label_list)):
        sys.exit()
        sys.exit(colored("ERROR: Image and labels list must have the same lenght!","red"))
    
    accuracies  = []
    confidences = []
    edistances  = []

    # start sliding window for every image
    for image,classid in zip(image_list,label_list):
        hDIM,wDIM  = image.size     
        img        = np.asarray(image)
        img        = scipy.misc.imresize(img, (hDIM,wDIM), interp="bicubic").astype(np.float32, casting='unsafe')
        img       -= scipy.ndimage.measurements.mean(img)       # confirmed. check data_utils.py on github
        img       /= np.std(img)                                # confirmed. check data_utils.py on github

        BLOCK = 8
        if(BLOCK > minimum):                        # checks if it isn't too big
            BLOCK = IMAGE                           # side of square block for painting: BLOCKxBLOCK. BLOCK <= IMAGE  
        
        padding   = (IMAGE - BLOCK) // 2            # padding for centering sliding window
        paddingh  = (HEIGHT - BLOCK) // 2           # padding for centering sliding window (rectangles)
        paddingw  = (WIDTH - BLOCK) // 2            # padding for centering sliding window (rectangles)

        nhDIM     = hDIM - 2*paddingh                                                
        nwDIM     = wDIM - 2*paddingw                                
        
        hshifts   = nhDIM // BLOCK                  # number of sliding window shifts on height
        wshifts   = nwDIM // BLOCK                  # number of sliding window shifts on width
        total = hshifts*wshifts                     # total number of windows
        counts = [0] * nclasses                     # will count the occurences of each class. resets at every image

        if(saveOutputImage):
            background = image
            segmented = Image.new('RGB', (wDIM,hDIM), "black")      # create mask for segmentation

        # sliding window (center)
        print("Classification started...")
        print("\t   Size: ", wDIM, "x", hDIM)
        print("\t  Block: ", BLOCK)

        label = np.zeros(nclasses)  # creates an empty array with lenght of number of classes
        label[classid] = 1          # create a class label by setting the value 1 on the corresponding index 

        val = 0     # will store the highest probability 
        ed = 0      # will store the sum of the euclidian distances
        # start measuring time
        start_time = time.time()
        for i in range(0,hshifts):
            h = i*BLOCK
            for j in range(0,wshifts):
                w = j*BLOCK
                img2 = img[h:h+HEIGHT,w:w+WIDTH]
                img2 = np.reshape(img2,(1,HEIGHT,WIDTH,3))
                #img2 = np.reshape(img2,(1,IMAGE,IMAGE))    # for RNN 

                probs = model.predict(img2)              # predicts image's classid

                prediction = np.asarray(probs[0])        # converts probabilities list to numpy array
                ed += np.linalg.norm(label-prediction)   # calculates euclidian distance

                index = np.argmax(probs)
                counts[index] = counts[index] + 1
                val += probs[0][index]

                # segment block
                if(saveOutputImage):
                    color = getColor(index)
                    
                    ki = h + paddingh               # to calculate only once
                    kf = h + paddingh + BLOCK       # to calculate only once
                    for k in range(ki,kf):
                        zi = w + paddingw           # to calculate only once per loop iteration
                        zf = w + paddingw + BLOCK   # to calculate only once per loop iterarion
                        for z in range(zi,zf):
                            try:
                                segmented.putpixel((z,k),color)
                            except:
                                print("segmentation")
                                pass
            
        # stop measuring time
        cls_time = time.time() - start_time
        print("\t   Time:  %.3f seconds" % cls_time)
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
            acc   = counts[classid] / total * 100   # accuracy per correct prediction (higher probability)
            most  = np.argmax(counts)               # most probable class
            print("Error check...")
            print("\t  Class: ", classid)
            print("\t  Total: ", total)
            print("\t Counts: ", counts)
            print("\tPredict: ", most)
            print("\t    Acc: ", acc, "%")
            print("Error checked!\n")

            error_file = "%s_error.txt" % runid
            ferror = open(error_file, "a+")

            array = ','.join(str(x) for x in counts)   # convert array of count into one single string
            ferror.write("Total: %5d | Class: %d | [%s] | Acc: %.3f | Time: %.3f\n" % (total,classid,array,acc,cls_time))
            ferror.close()

            # accuracy by counting correct predictions (highest probability)
            accuracies.append(acc)
            
            # accuracy value by the sum of the highest probabilities 
            # (not 100% sure when predicts wrong) 
            val = val / total
            confidences.append(val)

            # euclidian distances (should it divide by the number of images?)
            ed = ed / total
            edistances.append(ed)

    
    # round accuracy array to %.2f
    accuracies = np.around(accuracies,2)
    confidences = np.around(confidences,2)
    edistances = np.around(edistances,2)

    out_var = edistances
    avg_acc = sum(out_var) / len(image_list)

    return out_var,avg_acc,max(out_var),min(out_var)

# function that classifies a image by a sliding window using a local server 
def classify_local_server(model,ip,port,runid,nclasses):
    """
    Function that creates a local server and loads a model. It waits until
    another program makes a connection. It expects to receive a message 
    that contains the PATH to the image to classify and it label ID.

    Params:
        `model` - trained model
        `ip` - local server's IP
        `port` - local server's PORT
        `runid` - ID for this classification
        `nclasses` - number of classes

    Return: (not defined yet)
    """
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((ip, port))
    serversocket.listen(7) 
    print("Starting server '%s' on port %d...\n" % (ip,port))

    while True:
        connection, address = serversocket.accept()     # wait until it receives a message
        buf = connection.recv(512)                      # is it enough?
        buf = str(buf.decode('ascii'))                  # decode from byte to string

        buf = buf.split(" ")        # expects something like: "path_to_image class"
        filename = buf[0]           # get image path
        classid  = int(buf[1])      # get image class

        if(filename == "stop"):
            # ends the program
            break
        
        print("Filename: %s" % filename)    # just to confirm
        print("   Class: %d\n" % classid)   # just to confirm

        try:
            # tries to load image
            background = Image.open(filename)
            print(colored("SUCCESS: Loaded %s successfully!" % filename,"green"))
        except:
            # if it fails go to next iteration
            print(colored("ERROR: Couldn't open %s!" % filename,"red"))
            continue
        
        classify_sliding_window(model,background,classid,runid,nclasses)
        
    return None