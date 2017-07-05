from __future__ import division, print_function, absolute_import
import sys,os,platform,time,copy
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
from utils import architectures,dataset,classifier,helper
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
HEIGHT = 128
WIDTH  = 128

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
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv,_,_ = dataset.load_dataset_windows(traindir,HEIGHT,WIDTH,shuffled=True,validation=vs,
                                                                                            mean=False,gray=False,save_dd=True)
    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS
else:
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,_,_,_,_= dataset.load_dataset_windows(traindir,HEIGHT,WIDTH,shuffled=True,save_dd=False)

# load test images
# TIP: Use dataset.load_test_images or dataset.load_dataset_windows as X
Xt = Yt = []
Xt,Yt,_ = dataset.load_test_images(testdir,resize=(WIDTH,HEIGHT),mean=False,to_array=False,gray=False)
#Xt,Yt = dataset.load_test_images_from_index_file(testdir,"./dataset/signals/test/imgs_classes.txt")
testdir = None

#X = dataset.convert_images_colorspace(X,testdir)
#Xv = dataset.convert_images_colorspace(Xv,testdir)
#print("")
#testdir = None

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

# build network architecture
network,_ = architectures.build_network(arch,network,CLASSES)

# model definition
model = tflearn.DNN(network,checkpoint_path="models/%s" % run_id,tensorboard_dir='logs/',
                    max_checkpoints=None,tensorboard_verbose=0,best_val_accuracy=0.95,
                    best_checkpoint_path=None)  

# training parameters
EPOCHS = 500                        # maximum number of epochs 
SNAP = 5                            # evaluates network progress at each SNAP epochs
iterations = EPOCHS // SNAP         # number of iterations (or evaluations) 
use_criteria = True                 # use stop criteria
eval_criteria = 0.80                # evaluation criteria (confidence)

helper.print_net_parameters(bs=bs,vs=vs,epochs=EPOCHS,snap=SNAP,eval_criteria=eval_criteria,
                            use_criteria=use_criteria)

# creates a new accuracies' .csv
csv_filename = "%s_accuracies.txt" % run_id
helper.create_accuracy_csv_file(filename=csv_filename,testdir=testdir)

best_val_acc = 0
no_progress = 0
iteration_time = 0
total_training_time = 0

# training operation: can stop by reaching the max number of iterations OR Ctrl+C OR by not evolving
it = 0      
try:
    while(it < iterations):
        stime = time.time()
        train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria,X2=None)
        val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria,X2=None)
        test_acc = min_acc = None
        if(testdir is not None): 
            _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=run_id,
                                                                      printout=False,criteria=eval_criteria)
        
        ftime = time.time() - stime

        # save best model if there is a better validation accuracy
        if(val_acc > best_val_acc):
            no_progress = 0
            best_model = copy.copy(model)
            best_val_acc = val_acc
            print(colored("\nINFO: New best model!","yellow"))
        else:
            no_progress += 1

        # write to a .csv file the evaluation accuracy 
        helper.write_accuracy_on_csv(filename=csv_filename,train_acc=train_acc,val_acc=val_acc,
                                     test_acc=test_acc,min_acc=min_acc,time=total_training_time)
        # write accuracy's values on file
        helper.print_accuracy(name="Evaluation",train_acc=train_acc,val_acc=val_acc,test_acc=test_acc,
                              min_acc=min_acc,time=total_training_time,ctime=ftime)

        # NOTE: stop criteria check - accuracy AND no progress
        if(use_criteria and helper.check_stop_criteria(train_acc,val_acc,test_acc,99,no_progress,10)): break
        
        # repeats the training operation until it reaches one stop criteria
        iteration_time = time.time()
        model.fit(X,Y,n_epoch=SNAP,shuffle=True,show_metric=True,batch_size=bs,snapshot_step=False, 
                  snapshot_epoch=False,run_id=run_id,validation_set=(Xv,Yv),callbacks=None)
        
        iteration_time = time.time() - iteration_time
        total_training_time += iteration_time

        it += 1
# to stop the training at any moment by pressing Ctrl+C
except KeyboardInterrupt:
    # intermediate evaluation to check which is the best model once Ctrl+C was pressed
    train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
    val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)
    test_acc = min_acc = None
    if(testdir is not None): 
        _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=run_id,
                                                                  printout=False,criteria=eval_criteria)

# save best model (need this check if Ctr+C was pressed)
if(best_val_acc > val_acc):
    model = best_model
    print(colored("INFO: Restored the best model!","yellow"))

# save trained model
if(True):
    print("Saving trained model...")
    modelname = "models/%s.tflearn" % run_id
    print("\tModel: ",modelname)
    model.save(modelname)
    print("Trained model saved!\n")

# load model from saved file (optional)
if(False):
    print("Loading trained model...")  
    model.load("models/%s.tflearn" % run_id)
    print("\tModel: ","models/%s.tflearn" % run_id)
    print("Trained model loaded!\n")    

# final evaluation with the best model
stime = time.time()
train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)
test_acc = min_acc = None
if(testdir is not None): 
    _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=run_id,
                                                              printout=False,criteria=eval_criteria)

ftime = time.time() - stime
                             
# write to a .csv file the evaluation accuracy
helper.write_accuracy_on_csv(filename=csv_filename,train_acc=train_acc,val_acc=val_acc,
                             test_acc=test_acc,min_acc=min_acc,time=total_training_time)
# write accuracy's values on file
helper.print_accuracy(name="Final Eval",train_acc=train_acc,val_acc=val_acc,test_acc=test_acc,
                      min_acc=min_acc,time=None,ctime=ftime,color="green")

# NOTE: Turn show_image to false when scheduling many trainings
classifier.test_model_accuracy(model=model,image_set=Xv,label_set=Yv,
                               eval_criteria=eval_criteria,show_image=False)

# sound a beep to notify that the training ended
freq = 1000
dur  = 1500 
ws.Beep(freq,dur)   