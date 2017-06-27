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
import time,cv2
from utils import architectures,dataset,classifier
from colorama import init
from termcolor import colored

# init colored print
init()

if (len(sys.argv) < 5):
    print(colored("Call: $ python training.py {dataset} {architecture} {batch_size} {runid} [testdir]","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))

# clears screen and shows OS
classifier.clear_screen()

# NOTE: change if you want a specific size
HEIGHT = 32
WIDTH  = 32

# get command line arguments
traindir = sys.argv[1]         # path/to/cropped/images
arch     = sys.argv[2]         # name of architecture
bs       = int(sys.argv[3])    # batch size
run_id   = sys.argv[4]         # name for output model

try: 
    testdir = sys.argv[5]      # test images directory
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
Xt,Yt,mean_xte = dataset.load_test_images(testdir,resize=None,mean=False)
#Xt,Yt = dataset.load_test_images_from_index_file(testdir,"./dataset/signals/test/imgs_classes.txt")

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

# NOTE: an extra input_data layer. Use only when needed
#in2 = input_data(shape=[None,1])
#print(network.shape)
#print(in2.shape,"\n")
#network = architectures.build_merge_test(network,in2,CLASSES)

network,_ = architectures.build_network(arch,network,CLASSES)

# model definition
model = tflearn.DNN(network, checkpoint_path="models/%s" % run_id, tensorboard_dir='logs/',
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95,
                    best_checkpoint_path=None)  

# training parameters
EPOCHS = 2000                       # maximum number of epochs 
SNAP = 10                           # evaluates network progress at each SNAP epochs
iterations = EPOCHS // SNAP         # number of iterations (or evaluations) 
use_criteria = False                # use stop criteria
eval_criteria = 0.80                # evaluation criteria (confidence)

print("Batch size:", bs)
print("Validation:", vs)
print("    Epochs:", EPOCHS)
print("  Snapshot:", SNAP)
print("Eval crit.:", eval_criteria, "\n")

# creates a new accuracies' .csv
csv_file = "%s_accuracies.txt" % run_id
fcsv = open(csv_file,"w+")
if(testdir):
    fcsv.write("train,validation,test,min,time\n")
else:
    fcsv.write("train,validation,time\n")
fcsv.close()

best_train_acc = 0
best_val_acc = 0
best_test_acc = 0
time_per_epoch = 0
no_progress = 0

# training operation: can stop by reaching the max number of iterations or by Ctrl+C
# iterator to control the maximum number of iterations 
it = 0      
try:
    while(it < iterations):
        stime = time.time()
        train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria,X2=None)
        val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria,X2=None)

        test_acc = 0
        if(testdir): 
            _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=run_id,printout=False,criteria=eval_criteria,X2=None)
        
        ftime = time.time() - stime

        # save best model 
        # TODO: consider other accuracy as well
        if(val_acc > best_val_acc):
            no_progress = 0
            best_model = model
            best_val_acc = val_acc
            print(colored("New best model!","yellow"))

        # write to a .csv file the evaluation accuracy 
        fcsv = open(csv_file,"a+")
        if(testdir):
            fcsv.write("%.2f,%.2f,%.2f,%.2f,%.3f\n" % (train_acc,val_acc,test_acc,min_acc,time_per_epoch))
        else:
            fcsv.write("%.2f,%.2f,%.2f\n" % (train_acc,val_acc,time_per_epoch))
        fcsv.close()
        
        print(colored("\n======== Evaluation =========","yellow"))
        print("     Train:", train_acc, "%")
        print("Validation:", val_acc, "%")
        if(testdir): 
            print("      Test:", test_acc, "%")
            print("       Min:", min_acc, "%")
        print("Time/Epoch: %.3f seconds" % time_per_epoch)
        print(colored("=============================","yellow"))
        print(colored("Time: %.3f seconds\n" % ftime,"yellow"))
        
        # stop criteria by reaching a certain accuracy
        if(testdir):
            if(use_criteria and train_acc > 97.5 and val_acc > 97.5 and test_acc > 97.5):
                break
        else:
            if(use_criteria and train_acc > 97.5 and val_acc > 97.5):
                break
        
        # repeats the training operation until it reaches one stop criteria
        time_per_epoch = time.time()
        
        model.fit(X, Y, n_epoch=SNAP, shuffle=True, show_metric=True, 
                  batch_size=bs, snapshot_step=False, snapshot_epoch=False, 
                  run_id=run_id, validation_set=(Xv,Yv), callbacks=None)
        
        time_per_epoch = time.time() - time_per_epoch
        time_per_epoch = time_per_epoch / SNAP

        it += 1

# to stop the training at any moment by pressing Ctrl+C
except KeyboardInterrupt:
    pass 

fcsv.close()

# intermediate evaluation to check which is the best model
train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)
test_acc = 0
if(testdir): 
    _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=run_id,printout=False,criteria=eval_criteria)

# save best model
if(best_val_acc > val_acc):
    model = best_model
    print(colored("Restored the best model!","yellow"))

# save trained model
print("Saving trained model...")
modelname = "models/%s.tflearn" % run_id
print("\tModel: ",modelname)
model.save(modelname)
print("Trained model saved!\n")

# load model to figure out if there is something wrong 
if(False):
    print("Loading trained model...")  
    model.load("models/%s.tflearn" % run_id)
    print("\tModel: ","models/%s.tflearn" % run_id)
    print("Trained model loaded!\n")    

# final evaluation with the best model
stime = time.time()
train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)

test_acc = 0
if(testdir): 
    _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=run_id,printout=False,criteria=eval_criteria)

ftime = time.time() - stime

# write to a .csv file the evaluation accuracy
fcsv = open(csv_file,"a+")
if(testdir):
    fcsv.write("%.2f,%.2f,%.2f,%.2f,%.2f\n" % (train_acc,val_acc,test_acc,min_acc,time_per_epoch))
else:
    fcsv.write("%.2f,%.2f,%.2f\n" % (train_acc,val_acc,time_per_epoch))
fcsv.close()

print(colored("===== Final Evaluation ======","green"))
print("     Train:", train_acc, "%")
print("Validation:", val_acc, "%")
if(testdir):
    print("      Test:", test_acc, "%")
    print("       Min:", min_acc, "%") 
print(colored("=============================","green"))
print(colored("Time: %.3f seconds\n" % ftime,"green"))

# shows image and predicted class
# NOTE: Turn to false when scheduling many trainings
show_image = True   # flag to (not) show tested images
if(True):
    print(colored("INFO: Showing dataset performance","yellow"))
    len_Xv = len(Xv)
    bp = 0   # badly predicted counter 
    wp = 0   # well predicted counter (confidence > criteria) 
    
    for i in np.arange(0,len_Xv):
        # classify the digit
        probs = model.predict(Xv[np.newaxis, i])
        probs = np.asarray(probs)
        # sorted indexes by confidences 
        predictions = np.argsort(-probs,axis=1)[0]
        # top-2 predictions
        guesses = predictions[0:2]
        
        ci = int(guesses[0])
        confidence = probs[0][ci]

        # resize the image to 128 x 128 
        image = Xv[i]
        image = cv2.resize(image, (128, 128))

        # show the image and prediction of badly predicted cases
        if(guesses[0] != np.argmax(Yv[i])):
            bp += 1
            #if(confidence < eval_criteria): bpc += 1
            
            if(show_image):
                print("Predicted: {0}, Actual: {1}, Confidence: {2:3.2f}, Second guess: {3}".format(guesses[0], np.argmax(Yv[i]),confidence,guesses[1]))
                cv2.imshow("Test image", image)
                key = cv2.waitKey(0)

            if(key == 27):
                cv2.destroyWindow("Test image")
                show_image = False
        else:
            if(confidence > eval_criteria):
                wp +=1

    print(colored("INFO: %d badly predicted images in a total of %d" % (bp,len_Xv),"yellow"))
    print(colored("INFO: %d well predicted images (confidence > %.2f) in a total of %d" % (wp,eval_criteria,len_Xv),"yellow"))

# sound a beep
freq = 1000
dur  = 1500 
ws.Beep(freq,dur)   