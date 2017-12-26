from __future__ import division, print_function, absolute_import

import os
import sys
import time
import platform
import six
import socket
import math
import PIL
import cv2
import numpy as np 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn

from PIL import Image
from colorama import init
from termcolor import colored

# init colored print
init()

IMAGE  = 128
HEIGHT = 128
WIDTH  = 128
CHANNELS = 3

minimum = min(IMAGE, HEIGHT, WIDTH)

# control flags for extra features
saveOutputImage = True
showProgress    = False and saveOutputImage

# return a color according to the class
def getColor(x):
    return {
        0 : (255,0,0),     # red
        1 : (0,255,0),     # green
        2 : (0,0,255),     # blue
        3 : (255,255,0),   # yellow
        4 : (255,128,5),   # orange  
        5 : (255,20,147),  # pink
        6 : (0,255,255),   # cyan
        7 : (255,255,255), # white
        8 : (128,0,128),   # purple
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
def classify_single_image(model, image, label=None):
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

    image = np.reshape(image, (1, HEIGHT, WIDTH, 3))
    ctime = time.time()
    probs = model.predict(image)
    ctime = time.time() - ctime
    index = np.argmax(probs)
    prob  = probs[0][index] 
    
    if(isinstance(label, int)):
        print("    Label:", label)
    
    print("Predicted: %d (Prob: %.2f)\n" % (index, prob))
    print("Time: %.3f seconds" % ctime)
    
    return index, prob

# function that classifies a set of images and returns their labelIDs and confidence
def classify_set_of_images(model, images_list, runid, batch_size=128, labels_list=None, printout=False):
    """
    Function that classifies a set of images. If is passed a label list to confirm, 
    it shows the prediction and its confidence. Assuming that images have the 
    same dimensions as training images.

    Params:
        `model` - trained model
        `images_list` - set of images to be classified
        `labels_list` (optional) - set of labels of the corresponding images
        `runid` - classification run ID
        `batch_size` - 
        `printout` - 

    Return: Images' labelID and confidences
    """

    length      = len(images_list)                  # lenght of images list
    iterations  = math.ceil(length/batch_size)      # counter of how many iterations will be done
    pointer     = 0                                 # pointer to the current batch
    indexes     = [-1] * length                     # array to store predictions labels
    confidences = [-1] * length                     # array to store predictions confidences

    if(labels_list):
        if(len(labels_list) != length):
            sys.exit(colored("[ERROR] Images and labels lists must have the same lenght!", "red"))

    for image in images_list:
        image = np.reshape(image, (1, HEIGHT, WIDTH, 3))

    ctime = time.time()
    for its in range(iterations):
        sub_images_list = images_list[pointer:pointer+batch_size]
        sub_images_list = [np.asarray(img) for img in sub_images_list] # NOTE: check if this works
        probs = model.predict(sub_images_list)

        for vals in probs:
            index = np.argmax(vals)
            indexes.append(index)
            confidences.append(vals[index])
        
        pointer += batch_size
    ctime = time.time() - ctime
    
    if(labels_list and printout):
        out_file = "%s_predicts.txt" % runid
        of = open(out_file, "w+")
        of.write("Label | Predict | Confidence\n")
        of.close()

        of = open(out_file, "a+")
        of.write("----------------------------\n")
        for i in range(0, length):
            of.write("  %2d  |   %2d    |    %.2f\n" % (labels_list[i], indexes[i], confidences[i]))
        
        of.close()
    print("\nTime: %.3f seconds" % ctime)

    return indexes, confidences

# similiar function to the TFlearn's Evaluate
def my_evaluate(model, images_list, labels_list, batch_size=128, criteria=0.75, X2=None):
    """
    Costumized evaluation function. Uses the confidence (%) criteria confidence as 
    a constraint to confirm if that an image is correctly classified. Meant to
    be used on images with the same dimensions as the training images.

    Params:
        `model` - network trained model
        `images_list` - image set
        `labels_list` - labels set 
        `batch_size` - number  
        `criteria` - minimum confindence to declare a good classification
    
    Return: Accuracy (in percentage)
    """
    length     = len(images_list)                           # length of images list
    iterations = math.ceil(length/batch_size)               # counter of how many batches will be used
    pointer = 0                                             # batch number pointer
    labels_list = [np.argmax(elem) for elem in labels_list] # convert labels to a simpler representation
    wp = 0                                                  # counter for well predicted images
    counter = 0                                             # global counter 

    # images and labels lists must have the same lenght
    if(len(labels_list) != length): 
        sys.exit(colored("[ERROR] Images and labels lists must have the same length!", "red"))

    ctime = time.time()
    for its in range(iterations):
        sub_images_list = images_list[pointer:pointer+batch_size]   # get batch of images
        sub_labels_list = labels_list[pointer:pointer+batch_size]   # get batch of labels

        if(False):
            # to use when there is more then one input layer
            sub_x2 = X2[pointer:pointer+batch_size]
            probs = model.predict([sub_images_list, sub_x2])  # make predictions
        else:
            probs = model.predict(sub_images_list)

        # probabilities array and labels batch must have the same length
        if(len(probs) != len(sub_labels_list)):
            sys.exit(colored("[ERROR] Probs and sub labels lists must have the same length!", "red"))

        for vals, classid in zip(probs, sub_labels_list):
            index = np.argmax(vals)     # get the index of the most probable class
            val = vals[index]           # get the confidence of the predicted class
            if(index == classid and val >= criteria):
                wp += 1
            counter += 1
      
        pointer += batch_size
    ctime = time.time() - ctime
    
    if(counter != length):
        sys.exit(colored("[ERROR] Counter and length must be equal!", "red"))
    
    acc = wp / length * 100
    return np.round(acc, 2)

# function that classifies a image thorugh a sliding window
def classify_sliding_window(model, images_list, labels_list, nclasses, runid=None, printout=True, criteria=0.75, X2=None):
    """
    Function that classifies a set of images through a sliding window. In an extreme
    situation, it classifies just only one window. Its meant to be used on images with
    larger dimensions than those used on the training but it also works with training 
    and validation sets.

    Params:
        `model` - trained model variable
        `images_list` - list of images to be classified (already loaded)
        `labels_list` - list of labels of the corresponding images (use [-1, ...] for collages)
        `runid` - classification ID
        `nclasses` - number of classes
        `printout` - if False, it surpresses all prints by redirecting STDOUT
        `criteria` - minimum confidence to correctly classify an image 

    Return: Tuple containing an (array, mean_accuracy, max, min)
    """

    # verifies if it must surpress all prints
    if(printout == False):
        actual_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    if(len(images_list) != len(labels_list)):
        sys.exit(colored("[ERROR] Image and labels list must have the same lenght!", "red"))
    
    accuracies  = []
    confidences = []
    edistances  = []

    # start sliding window for every image
    # NOTE: Attention when it is used with MEAN
    for image, classid in zip(images_list, labels_list):
    #for image, classid, mean in zip(images_list, labels_list, X2):
        # special treatment for training and validation datasets
        if(isinstance(image, PIL.Image.Image)):
            wDIM, hDIM  = image.size
        else:
            hDIM = image.shape[0]
            wDIM = image.shape[1]
            classid = np.argmax(classid)

        img = np.array(image)
        img = scipy.misc.imresize(img, (hDIM, wDIM), interp="bicubic").astype(np.float32, casting='unsafe')
        
        BLOCK = 128
        if(BLOCK > minimum or BLOCK < 2):   # checks if it isn't too big
            BLOCK = IMAGE                   # side of square block for painting: BLOCKxBLOCK. BLOCK <= IMAGE  
        
        padding   = (IMAGE - BLOCK) // 2    # padding for centering sliding window
        paddingh  = (HEIGHT - BLOCK) // 2   # padding for centering sliding window (rectangles)
        paddingw  = (WIDTH - BLOCK) // 2    # padding for centering sliding window (rectangles)

        nhDIM     = hDIM - 2*paddingh                                                
        nwDIM     = wDIM - 2*paddingw                                
        
        hshifts   = nhDIM // BLOCK          # number of sliding window shifts on height
        wshifts   = nwDIM // BLOCK          # number of sliding window shifts on width
        total = hshifts*wshifts             # total number of windows
        counts = [0] * nclasses             # will count the occurences of each class. resets at every image

        if(saveOutputImage):
            background = image                                      # copy image to a background image variable
            segmented = Image.new('RGB', (wDIM, hDIM), "black")      # create mask for segmentation

        # sliding window (center)
        print("Classification started...")
        print("\t   Size: ", wDIM, "x", hDIM)
        print("\t  Block: ", BLOCK)

        #label = np.zeros(nclasses)  # creates an empty array with lenght of number of classes
        #label[classid] = 1          # create a class label by setting the value 1 on the corresponding index 
                                     # for example, with 4 classes and for index 2 -> [0 0 1 0]

        val = 0     # will store the highest probability 
        ed = 0      # will store the sum of the euclidian distances
        wp = 0      # will store the number of images well predicted (confidence > 0.75)

        # start measuring time
        start_time = time.time()
        for i in range(0, hshifts):
            h = i*BLOCK
            for j in range(0, wshifts):
                w = j*BLOCK
                img2 = img[h:h+HEIGHT, w:w+WIDTH]
                img2 = np.reshape(img2, (1, HEIGHT, WIDTH, CHANNELS))
                #img2 = np.reshape(img2, (1, IMAGE, IMAGE))    # for RNN 

                if(False):
                    # to use when there is more then one input layer
                    mean = np.array(mean)
                    mean = np.reshape(mean, (-1, 1))
                    probs = model.predict([img2, mean])      # predicts image's classid
                else:
                    probs = model.predict(img2)

                # NOTE: Euclidian distance
                #prediction = np.asarray(probs[0])        # converts probabilities list to numpy array
                #ed += np.linalg.norm(label-prediction)   # calculates euclidian distance

                index = np.argmax(probs)
                counts[index] = counts[index] + 1
                
                # NOTE: well predicted counter only increases when confidence > criteria
                if(index == classid):
                    confidence = probs[0][index]
                    if(confidence >= criteria):
                        wp += 1
                
                # NOTE: Sums correct predictions' confidences.
                #val += probs[0][classid] 

                # segment block
                if(saveOutputImage):
                    color = getColor(index)
                    
                    ki = h + paddingh               # to calculate only once
                    kf = h + paddingh + BLOCK       # to calculate only once
                    for k in range(ki, kf):
                        zi = w + paddingw           # to calculate only once per loop iteration
                        zf = w + paddingw + BLOCK   # to calculate only once per loop iterarion
                        for z in range(zi, zf):
                            try:
                                segmented.putpixel((z, k), color)
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
            
            output  = "%s_C%d.png" % (runid, classid)
            new_img.save(output, "PNG")        

        # Error check if not a collage
        if (classid != -1):
            acc  = counts[classid] / total * 100   # accuracy per correct prediction (higher probability)
            acc2 = wp / total * 100                # accuracy per correct prediction (confidence > 0.75) 
            most = np.argmax(counts)               # most probable class
            
            print("Error check...")
            print("\t  Class: ", classid)
            print("\t  Total: ", total)
            print("\t Counts: ", counts)
            print("\tPredict: ", most)
            print("\t    Acc: ", acc, "%")
            print("\t    Acc: ", acc2, "% (confidence > .75)")
            print("Error checked!\n")

            # only writes output to file if there is a run_id
            if(runid):
                error_file = "%s_error.txt" % runid
                ferror = open(error_file, "a+")
                array = ', '.join(str(x) for x in counts)   # convert array of count into one single string
                ferror.write("Total: %5d | Class: %d | [%s] | Acc: %.2f | Acc2: %.2f | Time: %.3f\n" % (total, classid, array, acc, acc2, cls_time))
                ferror.close()

            # accuracy by counting correct predictions (highest probability OR confidence > 0.75)
            accuracies.append(acc2)
            
            # NOTE: accuracy value by the sum of the highest probabilities
            #val = val / total
            #confidences.append(val)

            # NOTE: euclidian distances (should it divide by the number of images?)
            #ed = ed / total
            #edistances.append(ed)
    
    # round accuracy array to %.2f
    accuracies = np.around(accuracies, 2)
    #confidences = np.around(confidences, 2)
    #edistances = np.around(edistances, 2)

    out_var = accuracies
    avg_acc = sum(out_var) / len(images_list)
    avg_acc = np.round(avg_acc, 2)

    # gets stdout back to normal
    if(printout == False):
        sys.stdout = actual_stdout

    return out_var, avg_acc, max(out_var), min(out_var)

# function that classifies a image through a sliding window using a local server 
def classify_local_server(model, ip, port, runid, nclasses):
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
    print("Starting server '%s' on port %d...\n" % (ip, port))

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
            print(colored("[SUCCESS] Loaded %s successfully!" % filename, "green"))
        except:
            # if it fails go to next iteration
            print(colored("[ERROR] Couldn't open %s!" % filename, "red"))
            continue
        
        classify_sliding_window(model, background, classid, nclasses, runid=runid, printout=True, criteria=0.8)
        
    return None

# function to test a model's accuracy by showing the images where it gets wrong
def test_model_accuracy(model, image_set, label_set, eval_criteria, show_image=True, cmatrix=None):
    """
    Function to test a model's accuracy by showing the images where it 
    predicts wrong.

    Params:
        `image_set` - images set to be classified
        `label_set` - labels set respective to the images
        `eval_criteria` - evaluation criteria used in the training 
        `show_image` - flag to (not) show images
        `cmatrix` - flag to (not) generate confusion matrix as a run ID  
    """
    print(colored("[INFO] Showing dataset performance", "yellow"))
    len_is = len(image_set)    # length of the dataset that will be tested
    bp = 0                     # badly predicted counter 
    wp = 0                     # well predicted counter (confidence > criteria) 

    if(cmatrix is not None):
        fcsv = open(cmatrix + "_cmatrix.txt", "w+")
        fcsv.write("predicted, label\n")

    for i in np.arange(0, len_is):
        # classify the digit
        probs = model.predict(image_set[np.newaxis, i])
        probs = np.asarray(probs)
        # sorted indexes by confidences 
        predictions = np.argsort(-probs, axis=1)[0]
        # top-2 predictions
        guesses = predictions[0:2]
        
        ci = int(guesses[0])
        confidence = probs[0][ci]

        true_label = np.argmax(label_set[i])

        if(cmatrix is not None):
            fcsv = open(cmatrix + "_cmatrix.txt", "a+")
            fcsv.write("%d, %d\n" % (guesses[0], true_label))

        # resize the image to 128 x 128 
        image = image_set[i]
        image = cv2.resize(image, (128, 128))

        # show the image and prediction of badly predicted cases
        if(guesses[0] != true_label):
            bp += 1
            
            if(show_image):
                print("Predicted: {0:2d}, Actual: {1:2d}, Confidence: {2:3.3f}, Second guess: {3:2d}".format(int(guesses[0]), true_label, confidence, int(guesses[1])))
                cv2.putText(image, str(guesses[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
                cv2.imshow("Test image", image)
                key = cv2.waitKey(0)
                if(key == 27):
                    # pressed Esc
                    cv2.destroyWindow("Test image")
                    show_image = False
        else:
            if(confidence > eval_criteria):
                wp +=1

    if(cmatrix is not None):
        fcsv.close()

    print(colored("[INFO] %d badly predicted images in a total of %d (Error rate %.4f)" % (bp, len_is, bp/len_is), "yellow"))
    print(colored("[INFO] %d well predicted images (confidence > %.2f) in a total of %d (Acc. %.4f)" % (wp, eval_criteria, len_is, wp/len_is), "yellow"))
