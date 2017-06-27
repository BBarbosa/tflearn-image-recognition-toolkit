from __future__ import division, print_function, absolute_import
import sys, os, platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn
import tensorflow as tf
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected,flatten
from tflearn.data_utils import shuffle,featurewise_zero_center,featurewise_std_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import winsound as ws
import numpy as np
import time,cv2,glob,shutil
from utils import architectures,dataset,classifier
from colorama import init
from termcolor import colored

# init colored print
init()

if (len(sys.argv) < 4):
    print(colored("Call: $ python testing.py {dataset} {architecture} {model_path} [testdir]","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))

# clears screen and shows OS
classifier.clear_screen()

# NOTE: change if you want a specific size
HEIGHT = 32
WIDTH  = 32

# get command line arguments
traindir   = sys.argv[1]         # path/to/cropped/images
arch       = sys.argv[2]         # name of architecture
model_path = sys.argv[3]         # name for output model

try: 
    testdir = sys.argv[4]      # test images directory
except:
    testdir = None

vs = 0.3    # percentage of data for validation (set manually)

# load dataset and get image dimensions
if(vs and True):
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv,mean_xtr,mean_xv = dataset.load_dataset_windows(traindir,HEIGHT,WIDTH,shuffled=True,validation=vs,mean=False,gray=True)
    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS
else:
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,_,_,_,_= dataset.load_dataset_windows(traindir,HEIGHT,WIDTH,shuffled=True)

# load test images
Xt = Yt = None
#Xt,Yt,mean_xte = dataset.load_test_images(testdir,resize=None,mean=False)
#Xt,Yt = dataset.load_test_images_from_index_file(testdir,"./dataset/signals/test/imgs_classes.txt")
Xt,filenames = dataset.load_image_set_from_folder(testdir,(HEIGHT,WIDTH))

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
                     data_preprocessing=img_prep,              # NOTE: always check PP
                     data_augmentation=None)                   # NOTE: always check DA

network,_ = architectures.build_network(arch,network,CLASSES)

# model definition
model = tflearn.DNN(network, checkpoint_path=None, tensorboard_dir='logs/',
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95,
                    best_checkpoint_path=None)  

eval_criteria = 0.80        # evaluation criteria (confidence)
print("Eval crit.:", eval_criteria, "\n")

# load model to figure out if there is something wrong 
print("Loading trained model...")  
model.load(model_path)
print("\tModel: ",model_path)
print("Trained model loaded!\n")    

# final evaluation with the best model
stime = time.time()
train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)

if(testdir and Xt is not None and Yt is not None): 
    _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=run_id,printout=False,criteria=eval_criteria)

ftime = time.time() - stime

print(colored("===== Final Evaluation ======","green"))
print("     Train:", train_acc, "%")
print("Validation:", val_acc, "%")
if(testdir and Xt is not None and Yt is not None):
    print("      Test:", test_acc, "%")
    print("       Min:", min_acc, "%") 
print(colored("=============================","green"))
print(colored("Time: %.3f seconds\n" % ftime,"green"))

# shows image and predicted class
# NOTE: Turn to false when scheduling many trainings
print(colored("INFO: Showing dataset performance","yellow"))

# NOTE: Choose image set
image_set = Xv

len_is = len(image_set)
bp = 0                      # badly predicted counter
wp = 0                      # well predicted counter  
separate = False            # separates images with help of a trained model
show_image = True           # flag to (not) show tested images

#for i in np.random.choice(np.arange(0, len_is), size=(20,)):
for i in np.arange(0,len_is):
    # classify the digit
    probs = model.predict(image_set[np.newaxis, i])
    probs = np.asarray(probs)

    # sorted indexes by confidences 
    predictions = np.argsort(-probs,axis=1)[0]
    guesses = predictions[0:2]

    ci = int(guesses[0])
    confidence = probs[0][ci]

    # resize the image to 128 x 128
    image = image_set[i]
    image = cv2.resize(image, (128, 128))

    # NOTE: exceptional case -----------------------------------------------
    if(separate):
        print("Predicted: {0}, Confidence: {1:3.2f}, Second guess: {2} ({3}/{4})".format(guesses[0],confidence,guesses[1],i+1,len_is))
        # NOTE: Expecting something like "./samples\\0.jpg"
        current_image = filenames[i]
        src = current_image

        dest_file = current_image.split("\\")
        dest_file.reverse()
        dest_file = dest_file[0]

        if(False):
            cv2.imshow("Test image", image)
            key = cv2.waitKey(0)
        
            if(key == 13):
                # enter
                # NOTE: Always adapt the destination folder
                dest = "./dataset/numbers/augmented_v2/" + str(guesses[0]) + "/_" + dest_file
                shutil.copy(src,dest)
            elif(key > 47 and key < 58):
                # works for [0,9] but not for >9
                dest = "./dataset/numbers/augmented_v2/" + str(key-48) + "/_" + dest_file
                shutil.copy(src,dest)
                pass
            elif(key == 32):
                # space for skip
                pass
            elif(key == 27):
                # escape
                break
            else:
                pass
        else:
            dest = "./dataset/numbers/augmented_v3/" + str(guesses[0]) + "/_" + dest_file
            shutil.copy(src,dest)
    
    else:
        # show badly predicted images --------------------------------------
        if(guesses[0] != np.argmax(Yv[i])):
            bp += 1
            #if(confidence < eval_criteria): bpc += 1

            if(show_image):
                print("Predicted: {0}, Actual: {1}, Confidence: {2:3.3f}, Second guess: {3}".format(guesses[0], np.argmax(Yv[i]), confidence, guesses[1]))
                cv2.imshow("Test image", image)
                key = cv2.waitKey(0)

            if(key == 27):
                cv2.destroyWindow("Test image") 
                show_image = False
        
        else:
            if(confidence > eval_criteria):
                wp += 1

print(colored("INFO: %d badly predicted images in a total of %d" % (bp,len_is),"yellow"))
print(colored("INFO: %d well predicted images (confidence > %.2f) in a total of %d" % (wp,eval_criteria,len_is),"yellow"))

# sound a beep
freq = 1000
dur  = 1500 
ws.Beep(freq,dur)   