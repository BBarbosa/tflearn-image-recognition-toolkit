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
import time,cv2,glob,shutil,argparse
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
HEIGHT = 64
WIDTH  = 64

# get command line arguments
traindir   = sys.argv[1]         # path/to/cropped/images
arch       = sys.argv[2]         # name of architecture
model_path = sys.argv[3]         # name for output model

try: 
    testdir = sys.argv[4]      # test images directory
except:
    testdir = None

vs = 1    # percentage of data for validation (set manually)
"""
# load dataset and get image dimensions
if(vs and True):
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv,mean_xtr,mean_xv = dataset.load_dataset_windows(traindir,HEIGHT,WIDTH,shuffled=True,
                                                                                            validation=vs,mean=False,gray=False)
    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS
else:
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,_,_,_,_= dataset.load_dataset_windows(traindir,HEIGHT,WIDTH,shuffled=True)
"""  

# to load CIFAR-10 dataset and MNIST
if(False):
    print("Loading dataset (from directory)...")

    #CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv = dataset.load_cifar10_dataset(data_dir=traindir)
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv = dataset.load_mnist_dataset(data_dir=traindir)

    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS

    print("\t         Path:",traindir)
    print("\tShape (train):",X.shape,Y.shape)
    print("\tShape   (val):",Xv.shape,Yv.shape)
    print("Data loaded!\n")


# load test images
Xt = Yt = None
#Xt,Yt,mean_xte = dataset.load_test_images(testdir,resize=None,mean=False)
#Xt,Yt = dataset.load_test_images_from_index_file(testdir,"./dataset/signals/test/imgs_classes.txt")
Xt,filenames = dataset.load_image_set_from_folder(testdir,resize=(WIDTH,HEIGHT),extension="*.png")

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
#tflearn.init_graph(num_cores=4)

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
print("Eval crit.:", eval_criteria)
print("Validation:", vs*100 , "%\n")

# load model to figure out if there is something wrong 
print("Loading trained model...")  
model.load(model_path)
print("\tModel: ",model_path)
print("Trained model loaded!\n")    

# final evaluation with the best model
stime = time.time()
#train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)
if(testdir and Xt is not None and Yt is not None): 
    _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=run_id,printout=False,criteria=eval_criteria)

ftime = time.time() - stime

print(colored("===== Final Evaluation ======","green"))
#print("     Train:", train_acc, "%")
print("Validation:", val_acc, "%")
if(testdir and Xt is not None and Yt is not None):
    print("      Test:", test_acc, "%")
    print("       Min:", min_acc, "%") 
print(colored("=============================","green"))
print(colored("Time: %.3f seconds\n" % ftime,"green"))

# shows image and predicted class
# NOTE: Turn to false when scheduling many trainings
print(colored("[INFO] Showing dataset performance","yellow"))

# NOTE: Choose image set
image_set = Xv

len_is = len(image_set)     # lenght of test set
if(len_is < 1):
    sys.exit(colored("[INFO] Test set has no images!","yellow"))

bp = 0                      # badly predicted counter
wp = 0                      # well predicted counter  
separate = False            # separates images with help of a trained model
show_image = True           # flag to (not) show tested images
cmatrix = "mnist_mnist_r6" # NOTE: manually set by user

if(cmatrix is not None):
    fcsv = open(cmatrix + "_cmatrix.txt","w+")
    fcsv.write("predicted,label\n")

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

    true_label = np.argmax(Yv[i])

    if(cmatrix is not None):
        fcsv = open(cmatrix + "_cmatrix.txt","a+")
        fcsv.write("%d,%d\n" % (guesses[0],true_label))

    # resize the image to 128 x 128
    image = image_set[i]
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (HEIGHT, WIDTH))

    # NOTE: exceptional case -----------------------------------------------
    if(separate):
        print("Predicted: {0}, Confidence: {1:3.2f}, Second guess: {2} ({3}/{4})".format(guesses[0],confidence,guesses[1],i+1,len_is))
        # NOTE: Expecting something like "./samples\\0.jpg"
        current_image = filenames[i]
        src = current_image

        dest_file = current_image.split("\\")
        dest_file.reverse()
        dest_file = dest_file[0]

        dest_folder = "./dataset/numbers/augmented_v7/" # NOTE: set manually by user

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
            dest = dest_folder + str(guesses[0]) + "/_" + dest_file
            shutil.copy(src,dest)
    
    else:
        # show badly predicted images --------------------------------------
        if(guesses[0] != true_label):
            bp += 1

            if(show_image):
                print("Predicted: {0}, Actual: {1}, Confidence: {2:3.3f}, Second guess: {3}".format(guesses[0], np.argmax(Yv[i]), confidence, guesses[1]))
                rgb = np.fliplr(image.reshape(-1,CHANNELS)).reshape(image.shape)
                rgb = cv2.resize(rgb, (WIDTH*4,HEIGHT*4), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Test image", rgb)
                key = cv2.waitKey(0)

            if(key == 27):
                # pressed Esc
                cv2.destroyWindow("Test image") 
                show_image = False
        
        else:
            if(confidence > eval_criteria):
                wp += 1

if(cmatrix is not None):
    fcsv.close()

print(colored("[INFO] %d badly predicted images in a total of %d (Error rate %.4f)" % (bp,len_is,bp/len_is),"yellow"))
print(colored("[INFO] %d well predicted images (confidence > %.2f) in a total of %d (Acc. %.4f)" % (wp,eval_criteria,len_is,wp/len_is),"yellow"))

# sound a beep
freq = 1000
dur  = 1500 
ws.Beep(freq,dur)   