from __future__ import division, print_function, absolute_import
import sys, os, platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn
import tensorflow as tf
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.data_utils import shuffle, featurewise_zero_center, featurewise_std_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import winsound as ws
import numpy as np
import time,cv2,glob,shutil,argparse
from utils import architectures, dataset, classifier
from colorama import init
from termcolor import colored

# init colored print
init()

# argument parser
parser = argparse.ArgumentParser(description="Testing script for CNNs.",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("--data_dir", required=True, help="directory to the training data", type=str)
parser.add_argument("--arch", required=True, help="architecture name", type=str)
parser.add_argument("--model", required=True, help="run identifier (id) / model's path", type=str)
# optional arguments
parser.add_argument("--test_dir", required=False, help="path to test images", type=str)
parser.add_argument("--height", required=False, help="images height (default=64)", default=64, type=int)
parser.add_argument("--width", required=False, help="images width (default=64)", default=64, type=int)
parser.add_argument("--video", required=False, help="use video capture device/video")
parser.add_argument("--save", required=False, help="save output image (boolean)", type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
# parse arguments
args = parser.parse_args()
# print args
print(args,"\n")

# images properties
HEIGHT = args.height
WIDTH  = args.width

HEIGHT = 34
WIDTH  = 128

vs = 1    # percentage of data for validation (set manually)

# load dataset and get image dimensions
if(vs and True):
    CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, _, _ = dataset.load_dataset_windows(args.data_dir,HEIGHT,WIDTH,shuffled=True,
                                                                                        validation=vs,mean=False,gray=False)
    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS
else:
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,_,_,_,_= dataset.load_dataset_windows(args.data_dir,HEIGHT,WIDTH,shuffled=True)

# to load CIFAR-10 or MNIST dataset
if(False):
    print("Loading dataset (from directory)...")

    #CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv = dataset.load_cifar10_dataset(data_dir=args.data_dir)
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv = dataset.load_mnist_dataset(data_dir=args.data_dir)

    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS

    print("\t         Path:",args.data_dir)
    print("\tShape (train):",X.shape,Y.shape)
    print("\tShape   (val):",Xv.shape,Yv.shape)
    print("Data loaded!\n")


# load test images
Xt = Yt = None
#Xt,Yt,mean_xte = dataset.load_test_images(args.test_dir,resize=None,mean=False)
#Xt,Yt = dataset.load_test_images_from_index_file(args.test_dir,"./dataset/signals/test/imgs_classes.txt")
#Xt,filenames = dataset.load_image_set_from_folder(args.test_dir,resize=(WIDTH,HEIGHT),extension="*.png")
#_, Xt, Yt, _, _, _ , _, _, _, _ = dataset.load_dataset_windows(args.data_dir,HEIGHT,WIDTH,shuffled=True,
#                                                               validation=0,mean=False,gray=False)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
#img_prep.add_samplewise_zero_center(per_channel=True)   # per sample (featurewise is a global value)
img_prep.add_samplewise_zero_center()
img_prep.add_samplewise_stdnorm()       # per sample (featurewise is a global value)

# Real-time data augmentation
img_aug = ImageAugmentation()
#img_aug.add_random_flip_leftright()
#img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=10.)

# computational resources definition (made changes on TFLearn's config.py)
tflearn.init_graph(num_cores=8,allow_growth=True)
#tflearn.init_graph(num_cores=4)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS],    # shape=[None,IMAGE, IMAGE] for RNN
                     data_preprocessing=img_prep,              # NOTE: always check PP
                     data_augmentation=None)                   # NOTE: always check DA

network,_ = architectures.build_network(args.arch,network,CLASSES)

# model definition
model = tflearn.DNN(network, checkpoint_path=None, tensorboard_dir='logs/',
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95,
                    best_checkpoint_path=None)  

eval_criteria = 0.80        # evaluation criteria (confidence)
print("[INFO] Eval crit.:", eval_criteria)
print("[INFO] Validation:", vs*100 , "%\n")

# load model to figure out if there is something wrong 
print("[INFO] Loading trained model...")  
model.load(args.model)
print("[INFO] Model: ",args.model)
print("[INFO] Trained model loaded!\n")    

# final evaluation with the best model
stime = time.time()
#train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
val_acc  = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)
val_acc2 = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=0.1)
ftime = time.time() - stime

print(colored("===== Final Evaluation ======","green"))
#print("     Train:", train_acc, "%")
print("Validation:", val_acc, "%", "(Confidence > %.2f)" % eval_criteria)
print("Validation:", val_acc2, "%")
if(args.test_dir and Xt is not None and Yt is not None):
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
cmatrix = None # NOTE: manually set by user

if(cmatrix is not None):
    fcsv = open(cmatrix + "_test_cmatrix.txt","w+")
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
        fcsv = open(cmatrix + "_test_cmatrix.txt","a+")
        fcsv.write("%d,%d\n" % (guesses[0],true_label))

    # resize the image to 128 x 128
    image = image_set[i]
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (HEIGHT, WIDTH))

    # ///////////////// to separate images by its classes ///////////////// 
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
        # ///////////////// show badly predicted images /////////////////
        if(guesses[0] != true_label):
            bp += 1

            if(show_image):
                print("Predicted: {0:2d}, Actual: {1:2d}, Confidence: {2:3.3f}, Second guess: {3:2d}".format(int(guesses[0]), np.argmax(Yv[i]), confidence, int(guesses[1])))
                rgb = np.fliplr(image.reshape(-1,CHANNELS)).reshape(image.shape)
                rgb = cv2.resize(rgb, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)

                cv2.putText(rgb, str(guesses[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)

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