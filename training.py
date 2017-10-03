"""
Training script written in Tensorflow + TFLearn
"""

from __future__ import division, print_function, absolute_import
import sys,os,platform,time,argparse
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

# clears screen and shows OS
classifier.clear_screen()

# argument parser
parser = argparse.ArgumentParser(description="Hihg level Tensorflow training script.",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("train_dir",help="directory to the training data",type=str)
parser.add_argument("architecture",help="architecture name",type=str)
parser.add_argument("batch_size",help="batch size",type=int)
parser.add_argument("run_id",help="model's path",type=str)
# optional arguments
parser.add_argument("--test_dir",help="directory to the training data",type=str)
parser.add_argument("--height",help="images height (default 32)",default=32,type=int)
parser.add_argument("--width", help="images width (default 32)", default=32,type=int)
parser.add_argument("--val_set",help="percentage of training data to validation (default 0.3)",default=0.3,type=float)
parser.add_argument("--gray",help="convert images to grayscale",default=False,type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--freeze",help="freeze graph (not for retraining)",default=False,type=lambda s: s.lower() in ['true', 't', 'yes', '1'])

# parse arguments
args = parser.parse_args()
print(args,"\n")

# images properties
HEIGHT = args.height
WIDTH  = args.width

# load dataset and get image dimensions
if(args.val_set and True):
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,Xv,Yv,_,_ = dataset.load_dataset_windows(args.train_dir,HEIGHT,WIDTH,shuffled=True,validation=args.val_set,
                                                                               mean=False,gray=args.gray,save_dd=False)
    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS
else:
    CLASSES,X,Y,HEIGHT,WIDTH,CHANNELS,_,_,_,_= dataset.load_dataset_windows(args.train_dir,HEIGHT,WIDTH,shuffled=True,save_dd=False)
    Xv = Yv = []

# load test images
# NOTE: Use dataset.load_test_images or dataset.load_dataset_windows like on X, Y, Xv and Yv
if(args.test_dir is not None):
    _,Xt,Yt,_,_,_,_,_,_,_ = dataset.load_dataset_windows(args.test_dir,HEIGHT,WIDTH,shuffled=True,mean=False,gray=args.gray,save_dd=True)
    #Xt,Yt,_ = dataset.load_test_images(args.test_dir,resize=(WIDTH,HEIGHT),mean=False,to_array=False,gray=False)
    #Xt,Yt = dataset.load_test_images_from_index_file(args.test_dir,"./dataset/signals/test/imgs_classes.txt")
    #X = dataset.convert_images_colorspace(images_array=X,fromc=None,convert_to=args.test_dir)
    #Xv = dataset.convert_images_colorspace(images_array=Xv,fromc=None,convert_to=args.test_dir)

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
network,_ = architectures.build_network(args.architecture,network,CLASSES)

# model definition
model = tflearn.DNN(network,checkpoint_path="models/%s" % args.run_id,tensorboard_dir="logs/",
                    max_checkpoints=None,tensorboard_verbose=0,best_val_accuracy=0.95,
                    best_checkpoint_path=None)  

# training parameters
EPOCHS = 500                    # maximum number of epochs 
SNAP = 5                        # evaluates network progress at each SNAP epochs
iterations = EPOCHS // SNAP     # number of iterations (or evaluations) 
use_criteria = True             # use stop criteria
eval_criteria = 0.80            # evaluation criteria (confidence)

best_val_acc = 0                # best validation accuracy 
no_progress = 0                 # counter of how many snapshots without learning process
iteration_time = 0              # time between each snapshot
total_training_time = 0         # total training time

# print networks parameters on screen
helper.print_net_parameters(bs=args.batch_size,vs=args.val_set,epochs=EPOCHS,snap=SNAP,eval_criteria=eval_criteria,
                            use_criteria=use_criteria)

# creates a new accuracies file.csv
csv_filename = "%s_accuracies.txt" % args.run_id
helper.create_accuracy_csv_file(filename=csv_filename,testdir=args.test_dir,traindir=args.train_dir,vs=args.val_set,
                                height=HEIGHT,width=WIDTH,arch=args.architecture,bs=args.batch_size,epochs=EPOCHS,ec=eval_criteria)

# training operation: can stop by reaching the max number of iterations OR Ctrl+C OR by not evolving
it = 0      
try:
    while(it < iterations):
        stime = time.time()
        train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
        val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)
        
        test_acc = min_acc = None
        if(args.test_dir is not None): 
            _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=args.run_id,
                                                                      printout=False,criteria=eval_criteria)
            test_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=0.1)
        
        ftime = time.time() - stime

        # save best model if there is a better validation accuracy
        if(val_acc > best_val_acc):
            no_progress = 0
            best_val_acc = val_acc
            # Save best model to file
            if(args.freeze):
                # NOTE: use it for freezing model
                del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:] 
            
            print(colored("\n[INFO] New best model!","yellow"))
            print(colored("[INFO] Saving best trained model soo far...","yellow"))
            modelname = "models/%s.tflearn" % args.run_id
            print(colored("[INFO] Model: %s" % modelname,"yellow"))
            model.save(modelname)
            print(colored("[INFO] Best trained model saved!\n","yellow"))
            new_best = True    
        else:
            no_progress += 1
            new_best = False

        # write to a .csv file the evaluation accuracy 
        helper.write_accuracy_on_csv(filename=csv_filename,train_acc=train_acc,val_acc=val_acc,test_acc=test_acc,
                                     min_acc=min_acc,time=total_training_time,best=new_best)
        # write accuracy's values on file
        helper.print_accuracy(name="Evaluation",train_acc=train_acc,val_acc=val_acc,test_acc=test_acc,
                              min_acc=min_acc,time=total_training_time,ctime=ftime)

        # NOTE: stop criteria check - accuracy AND no progress
        if(use_criteria and helper.check_stop_criteria(train_acc,val_acc,test_acc,99,no_progress,10)): break
        
        # repeats the training operation until it reaches one stop criteria
        iteration_time = time.time()
        model.fit(X,Y,n_epoch=SNAP,shuffle=True,show_metric=True,batch_size=args.batch_size,snapshot_step=False, 
                  snapshot_epoch=False,run_id=args.run_id,validation_set=(Xv,Yv),callbacks=None)
        
        iteration_time = time.time() - iteration_time
        total_training_time += iteration_time

        it += 1
# to stop the training at any moment by pressing Ctrl+C
except KeyboardInterrupt:
    # intermediate evaluation to check which is the best model once Ctrl+C was pressed
    train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
    val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)
    
    test_acc = min_acc = None
    if(args.test_dir is not None): 
        _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=args.run_id,
                                                                  printout=False,criteria=eval_criteria)
        test_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=0.1)

# load best model (need this check if Ctr+C was pressed)
if(best_val_acc > val_acc):
    print(colored("[INFO] Loading best trained model...","yellow"))  
    model.load("models/%s.tflearn" % args.run_id)
    print("Model: models/%s.tflearn" % args.run_id)
    print(colored("[INFO] Restored the best model!","yellow"))
else:
    # save actual trained model
    if(args.freeze):
        # NOTE: use it for freezing model
        del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]

    print(colored("[INFO] Saving trained model...","yellow"))
    modelname = "models/%s.tflearn" % args.run_id
    print(colored("[INFO] Model: %s" % modelname,"yellow"))
    model.save(modelname)
    print(colored("[INFO] Trained model saved!\n","yellow"))    

# final evaluation with the best model
stime = time.time()
train_acc = classifier.my_evaluate(model,X,Y,batch_size=128,criteria=eval_criteria)
val_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=eval_criteria)

test_acc = min_acc = None
if(args.test_dir is not None): 
    _,test_acc,_,min_acc = classifier.classify_sliding_window(model,Xt,Yt,CLASSES,runid=args.run_id,
                                                              printout=False,criteria=eval_criteria)

    test_acc = classifier.my_evaluate(model,Xv,Yv,batch_size=128,criteria=0.1)

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