from __future__ import division, print_function, absolute_import

import os,sys,time,platform,six
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflearn
import winsound as ws  # for windows only
import tensorflow as tf
import scipy.ndimage
import socket

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
    print(colored("Call: $ python classify_sv.py {ip} {port} {architecture} {model}","red"))
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
    HEIGHT  = 128   
    WIDTH   = 128
    classes = 7

    minimum = min(IMAGE, HEIGHT, WIDTH)

    # get command line arguments
    ip        = sys.argv[1]       # server IP
    port      = int(sys.argv[2])  # port
    arch      = sys.argv[3]       # name of architecture
    modelpath = sys.argv[4]       # path to saved model

    # a bunch of flags
    saveOutputImage = False
    showProgress    = False and saveOutputImage # required saveOutputImage flag to show the progress

    # computational resources definition
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list=""
    config.log_device_placement=True
    #tf.add_to_collection('graph_config', config)

    #tflearn.init_graph(num_cores=8,gpu_memory_fraction=0.2)

    # network definition
    network = input_data(shape=[None, HEIGHT, WIDTH, 3],     # shape=[None,IMAGE, IMAGE] for RNN
                        data_preprocessing=None,       
                        data_augmentation=None) 

    network = architectures.build_network(arch,network,classes)

    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables()) 
    # model definition
    model = tflearn.DNN(network, checkpoint_path='models', session=sess,
                        max_checkpoints=1, tensorboard_verbose=0) # tensorboard_dir='logs'

    print("Loading trained model...")  
    model.load(modelpath)
    print("\tModel: ",modelpath)
    print("Trained model loaded!\n")
    
    #------------------------------ creates a local server ------------------------------
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((ip, port))
    serversocket.listen(7) # become a server socket, maximum 1 connections
    print("Starting server '%s' on port %d...\n" % (ip,port))

    while True: 
        connection, address = serversocket.accept()     # wait until it receives a message

        sttime = time.time() # start measuring total time

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
    
    #-------------------------------------------------------------------------------------

        wDIM,hDIM  = background.size     
        img        = scipy.ndimage.imread(filename, mode='RGB')     # mode='L', flatten=True -> grayscale
        img        = scipy.misc.imresize(img, (hDIM,wDIM), interp="bicubic").astype(np.float32, casting='unsafe')
        img       -= scipy.ndimage.measurements.mean(img)           # check data_utils.py on tflearn's github
        img       /= np.std(img)                                    # check data_utils.py on tflearn's github
    
        BLOCK = 8                                              # side of square block for painting: BLOCKxBLOCK
        if(BLOCK > minimum):                                        # checks if it isn't too big
            BLOCK = IMAGE

        padding   = (IMAGE - BLOCK) // 2                            # padding for centering sliding window
        paddingh  = (HEIGHT - BLOCK) // 2                           # padding for centering sliding window (rectangles)
        paddingw  = (WIDTH - BLOCK) // 2                            # padding for centering sliding window (rectangles)
        
        nhDIM     = hDIM - 2*(paddingh)                             # calculates the new dimension that restricts the sliding window 
        nwDIM     = wDIM - 2*(paddingw)                             # calculates the new dimension that restricts the sliding window 
        
        hshifts   = nhDIM // BLOCK                                  # number of sliding window shifts on height
        wshifts   = nwDIM // BLOCK                                  # number of sliding window shifts on width
        total     = hshifts*wshifts                                 # total number of windows
        segmented = Image.new('RGB', (wDIM,hDIM), "black")          # create mask for segmentation. same dimension as background

        counts = [0] * classes                                      # will count the occurences of each class

        # to check dims
        if(False):
            print("  height:",hDIM)
            print("   width:",wDIM)
            print(" padding:",padding)
            print("hpadding:",paddingh)
            print("wpadding:",paddingw)
            print("   nhDIM:",nhDIM)
            print("   nwDIM:",nwDIM)
            print(" hshifts:",hshifts)
            print(" wshifts:",wshifts)

        # sliding window (center)
        print("Classification started...")
        print("\t  Image: ", filename)
        print("\t   Size: ", wDIM, "x", hDIM)
        print("\t  Block: ", BLOCK)
    
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

                try:
                    img2 = np.reshape(img2,(1,HEIGHT,WIDTH,3))
                    #img2 = np.reshape(img2,(1,IMAGE,IMAGE))    # for RNN
                except:
                    print("reshape")
                    pass

                with tf.device('/cpu:0'):    
                    probs = model.predict(img2)
                
                index = np.argmax(probs)
                counts[index] = counts[index] + 1

                # to show the probability of every sample sent to the network 
                if(False):
                    val = probs[0][index]
                    print("Pixel %3d %3d | Class %d | Probability %f" % (h,w,index,val))
                    print("Probablilities ", probs,"\n")
                    input("Press any key to continue...\n")    

                # segment block
                if(saveOutputImage):
                    color = getColor(index)
                    
                    ki = h + paddingh                   # to calculate only once
                    kf = h + paddingh + BLOCK           # to calculate only once
                    for k in range(ki,kf):
                            zi = w + paddingw           # to calculate only once per loop iteration
                            zf = w + paddingw + BLOCK   # to calculate only once per loop iterarion
                            for z in range(zi,zf):
                                try:
                                    segmented.putpixel((z,k),color)
                                except:
                                    print("segmentation")
                                    pass

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
        cls_time = time.time() - start_time
        print("\t  Total:  %d images" % total)
        print("\t   Time:  %.3f seconds" % cls_time)
        print("Classification done!\n")

        # assuming modelpath: "models\epochs\name.tflearn" -> name
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
                test_id = filename.split('\\')[3].split('_')[0] 
                output  = "%s_%s_C%d_%s.png" % (modelname,arch,classid,test_id) # allows many test images per class 
            else:
                output  = "%s_%s.png" % (modelname,arch) # works for 1 test image per class too
    
            new_img.save(output,"png")
        
        # Error check if not a collage or a multiclass image
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
            ferror.write("Total: %5d | Class: %d | [%s] | Acc: %.2f | File: %s | Time: %f\n" % (total,classid,array,acc,filename,cls_time))
            ferror.close()

        if(showProgress):
            input("Press any key to continue...\n")    

        if(OS == 'Windows' and False):
            freq = 2000
            dur  = 1000 
            ws.Beep(freq,dur)
        
        fttime = time.time() - sttime       # stops measuring total time
        print(colored("INFO: Total time: %.3f\n" % fttime,"yellow"))
    
    connection.close()