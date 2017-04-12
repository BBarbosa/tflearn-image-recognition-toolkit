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

# script arguments' check
if(len(sys.argv) < 5):
    print(colored("Call: $ python classify_sw.py {architecture} {model} {image} {classid}","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))
else:
    # specify OS
    OS = platform.system() 
    
    # clear screen and show OS
    if(OS == 'Windows'):
        os.system('cls')
    else:
        os.system('clear')
    print("Operating System: %s\n" % OS)

    # images properties (inherit from trainning?)
    IMAGE   = 128   
    HEIGHT  = IMAGE
    WIDTH   = HEIGHT
    classes = 7

    # get command line arguments
    arch      = sys.argv[1]       # name of architecture
    modelpath = sys.argv[2]       # path to saved model
    filename  = sys.argv[3]       # test image name/path
    classid   = int(sys.argv[4])  # test image class id. -1 for collages

    # a bunch of flags
    saveOutputImage = False
    showProgress    = False
    showConvolution = False

    # computational resources definition
    tflearn.init_graph(num_cores=8,gpu_memory_fraction=0.9)

    # network definition
    network = input_data(shape=[None, HEIGHT, WIDTH, 3],     # shape=[None,IMAGE, IMAGE] for RNN
                        data_preprocessing=None,       
                        data_augmentation=None) 

    network = architectures.build_network(arch,network,classes)

    # model definition
    model = tflearn.DNN(network, checkpoint_path='models',
                        max_checkpoints=1, tensorboard_verbose=0) # tensorboard_dir='logs'

    print("Loading trained model...")  
    model.load(modelpath)
    print("\tModel: ",modelpath)
    print("Trained model loaded!\n")

    # display convolutions in a figure
    if(showConvolution):
        layer = "Conv2D"
        display_convolutions(model,layer,padding=4,filename='',nrows=4)

    # Load the image file (need pre-processment)
    background = Image.open(filename)
    wDIM,hDIM  = background.size     
    img        = scipy.ndimage.imread(filename, mode='RGB')     # mode='L', flatten=True -> grayscale
    img        = scipy.misc.imresize(img, (wDIM,hDIM), interp="bicubic").astype(np.float32, casting='unsafe')
    img       -= scipy.ndimage.measurements.mean(img)           # confirmed. check data_utils.py on github
    img       /= np.std(img)                                    # confirmed. check data_utils.py on github
    
    # select the minimum side
    if(wDIM < hDIM):
        minimun = wDIM
    else:
        minimun = hDIM

    # ensures that test image is a square (checked)
    img = img[0:minimun,0:minimun]
    # do the same for the background image
    background = background.crop((0,0,minimun,minimun))
    
    BLOCK     = 128                                               # side of square block for painting: BLOCKxBLOCK. Assume BLOCK <= IMAGE
    padding   = (IMAGE - BLOCK) // 2                            # padding for centering sliding window    
    nhDIM     = minimun - 2*padding
    nwDIM     = minimun - 2*padding
    hshifts   = nhDIM // BLOCK                                  # number of sliding window shifts on height
    wshifts   = nwDIM // BLOCK                                  # number of sliding window shifts on width
    segmented = Image.new('RGB', (minimun,minimun), "black")    # create mask for segmentation
        
    counts = [0] * classes                                      # will count the occurences of each class

    # to check dims
    if(False):
        print("   Side:",minimun)
        print("Padding:",padding)
        print("  nhDIM:",nhDIM)
        print("  nwDIM:",nwDIM)
        print("hshifts:",hshifts)
        print("wshifts:",wshifts)
        exit(1)

    # sliding window (center)
    print("Classification started...")
    print("\t  Image: ", filename)
    print("\t   Size: ", wDIM, "x", hDIM)
    print("\t  Block: ", BLOCK)
    print("\tResized: ", minimun, "x", minimun)
    
    # show progress
    if(showProgress):
        fig = plt.figure()
        ax  = fig.gca() 
        fig.show()

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
            
            # to show the probability of every sample sent to the network 
            if(False):
                print("Pixel %3d %3d | Class %d | Probability %f" % (h,w,index,val))
                print("Probablilities ", probs,"\n")

            # segment block
            if(saveOutputImage):
                for k in range(h+padding,h+BLOCK+padding):
                        for z in range(w+padding,w+BLOCK+padding):
                            segmented.putpixel((z,k),color)

        # show progress at each 3 lines
        if(showProgress and (i%3 == 0)):
            background = background.convert("RGBA")
            segmented = segmented.convert("RGBA")
            new_img = Image.blend(background, segmented, 0.3)           
            
            ax.imshow(new_img)
            ax.axis('off') 
            fig.canvas.draw()
            plt.pause(0.0001)
    
    # stop measuring time
    seg_time = time.time() - start_time
    print("\t   Time:  %s seconds" % seg_time)
    print("Classification done!\n")

    # assuming modelpath: "models\name.tflearn" -> name
    try:
        modelname = modelpath.split("\\")[1].split(".")[0]
    except:
         modelname = modelpath.split("/")[1].split(".")[0]

    # save output image options
    if(saveOutputImage):
        background = background.convert("RGBA")
        segmented = segmented.convert("RGBA")
        new_img = Image.blend(background, segmented, 0.3)

        # for a test image identifies 
        if (classid != -1):
            # assuming: "dataset\\fabric\\test_r\\c1\\A8_1_test.jpg" -> A8
            test_id = filename.split('\\')[4].split('_')[0] 
            output  = "%s_%s_%d_C%d_%s.png" % (modelname,arch,minimun,classid,test_id) # allows many test images per class 
        else:
            output  = "%s_%s_%d_C%d.png" % (modelname,arch,minimun,classid) # works for 1 test image per class too
    
        new_img.save(output,"PNG")
        
    # Error check if not a collage
    if (classid != -1):
        total = hshifts*wshifts                 # total number of windows
        acc   = counts[classid] / total * 100   # accuracy
        most  = np.argmax(counts)               # most probable class
        print("Error check...")
        print("\t  Image: ", filename)
        print("\t  Class: ", classid)
        print("\t  Total: ", total)
        print("\t Counts: ", counts)
        print("\tPredict: ", most)
        print("\t    Acc: ", acc, "%")
        print("Error checked!\n")

        error_file = "%s_%s_error.txt" % (modelname,arch)
        ferror = open(error_file, "a+")
                 
        array = ','.join(str(x) for x in counts)   # convert array of count into one single string
        ferror.write("Total: %5d | Class: %d | [%s] | Acc: %.2f | File: %s | Time: %f\n" % (total,classid,array,acc,filename,seg_time))
        ferror.close()

    if(showProgress):
        input("Press any key to continue...\n")    

    if(OS == 'Windows'):
        freq = 2000
        dur  = 1000 
        ws.Beep(freq,dur)
                
