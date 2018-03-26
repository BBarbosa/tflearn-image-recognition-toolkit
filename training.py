"""
Training script written in Tensorflow and TFLearn
for image classification.

Author: bbarbosa
"""

from __future__ import division, print_function, absolute_import

import os
import sys
import time
import platform
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn
import tensorflow as tf

import tflearn.helpers.summarizer as s  
from tflearn.layers.core import input_data
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from utils import architectures, dataset, classifier, helper, plot    
from colorama import init
from termcolor import colored

# init colored print
init()

true_cases = ['true', 't', 'yes', '1']

# argument parser
custom_formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=2000)
parser = argparse.ArgumentParser(description="High level Tensorflow and TFLearn training script.", 
                                 prefix_chars='-',
                                 formatter_class=custom_formatter_class)
# required arguments
parser.add_argument("--train_dir", required=True, help="<REQUIRED> directory to the training data", type=str)
parser.add_argument("--arch", required=True, help="<REQUIRED> architecture name", type=str)
parser.add_argument("--model_name", required=True, help="<REQUIRED> Model name / Path to trained model", type=str)
# optional arguments
parser.add_argument("--bsize", required=False, help="batch size (default=16)", default=16, type=int)
parser.add_argument("--test_dir", required=False, help="directory to the testing data (default=None)", type=str)
parser.add_argument("--height", required=False, help="images height (default=64)", default=64, type=int)
parser.add_argument("--width", required=False, help="images width (default=64)", default=64, type=int)
parser.add_argument("--val_set", required=False, help="percentage of training data to validation (default=0.3)", default=0.3, type=float)
parser.add_argument("--gray", required=False, help="convert images to grayscale (default=False)", default=False, type=lambda s: s.lower() in true_cases)
parser.add_argument("--freeze", required=False, help="freeze graph (not for retraining) (default=False)", default=False, type=lambda s: s.lower() in true_cases)
parser.add_argument("--snap", required=False, help="evaluate training frequency (default=5)", default=5, type=int)
parser.add_argument("--pproc", required=False, help="enable/disable pre-processing (default=True)", default=True, type=lambda s: s.lower() in true_cases)
parser.add_argument("--aug", required=False, nargs="+", help="enable data augmentation (default=[])", default=[])
parser.add_argument("--n_epochs", required=False, help="maximum number of training epochs (default=1000)", default=1000, type=int)
parser.add_argument("--eval_crit", required=False, help="classification confidence threshold (default=0.75)", default=0.75, type=float)
parser.add_argument("--cspace", required=False, help="convert images color space (default=None)", default=None, type=str)
parser.add_argument("--param", required=False, help="versatile extra parameter (default=None)", default=None)
parser.add_argument("--show", required=False, help="show test model & PDF report (default=False)", default=False, type=lambda s: s.lower() in true_cases)

# parse arguments
args = parser.parse_args()

print(args, "\n")

# images properties
HEIGHT = args.height
WIDTH  = args.width

""" """
# load dataset and get image dimensions
if(args.val_set):
    CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, _, _ = dataset.load_dataset_windows(args.train_dir, HEIGHT, WIDTH, shuffled=True, 
                                                                                        validation=args.val_set, gray=args.gray, 
                                                                                        save_dd=False, data_aug=args.aug)
    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS
else:
    CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, _, _, _, _= dataset.load_dataset_windows(args.train_dir, HEIGHT, WIDTH, shuffled=True,
                                                                                     save_dd=False, data_aug=args.aug)
    Xv = Yv = []

# load test images
if(args.test_dir is not None):
    _, Xt, Yt, _, _, _, _, _, _, _ = dataset.load_dataset_windows(args.test_dir, HEIGHT, WIDTH, shuffled=True, mean=False, 
                                                                  gray=args.gray, save_dd=True, data_aug=args.aug)
 

# to load CIFAR-10 dataset or MNIST
if(False):
    print("[INFO] Loading dataset (from directory)...")

    CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, Xt, Yt = dataset.load_mnist_dataset(train_dir=args.train_dir)
    #CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, Xt, Yt = dataset.load_cifar10_dataset(train_dir=args.train_dir)

    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS

    print("[INFO] \t         Path:", args.train_dir)
    print("[INFO] \tShape (train):", X.shape, Y.shape)
    print("[INFO] \tShape   (val):", Xv.shape, Yv.shape)
    print("[INFO] Data loaded!\n")

# to convert images color spaces
if(args.cspace is not None):
    X, CHANNELS  = dataset.convert_images_colorspace(images_array=X, fromc=None, convert_to=args.cspace)
    Xv, CHANNELS = dataset.convert_images_colorspace(images_array=Xv, fromc=None, convert_to=args.cspace)
    if(args.test_dir is not None):
        Xt, CHANNELS = dataset.convert_images_colorspace(images_array=Xt, fromc=None, convert_to=args.cspace)
    
    classifier.CHANNELS = CHANNELS

# Real-time data preprocessing (samplewise or featurewise)
img_prep = None
if(args.pproc):
    img_prep = ImagePreprocessing()
    img_prep.add_samplewise_zero_center()
    img_prep.add_samplewise_stdnorm() 
    img_prep.add_crop_center(shape=(HEIGHT, WIDTH))     
    #img_prep.add_zca_whitening()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=5.)

# computational resources definition (made changes on TFLearn's config.py)
tflearn.init_graph(num_cores=8, allow_growth=True)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS], 
                     data_preprocessing=img_prep,           
                     data_augmentation=None)                

# build network architecture
network, _ = architectures.build_network(name=args.arch, network=network, classes=CLASSES, 
                                         param=args.param)

# ////////////////////////////////////////////////////
#                Model definition
# ////////////////////////////////////////////////////
retraining = False
training_id = args.model_name

if(os.path.isfile(args.model_name + ".data-00000-of-00001")):
    retraining = True
    parts = args.model_name.split(os.sep)
    parts.reverse()
    training_id = parts[0].split('.')[0]

model = tflearn.DNN(network, checkpoint_path="./models/%s" % training_id, tensorboard_dir="./logs/", 
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95, 
                    best_checkpoint_path=None)

# training parameters
iterations = args.n_epochs // args.snap     # number of iterations (or evaluations)
use_criteria = True                         # use stop criteria

best_val_acc  = 0    # best validation accuracy (75% confidence)
best_test_acc = 0    # best test accuracy (normal)

no_progress         = 0     # counter of how many snapshots without learning process
iteration_time      = 0     # time between each snapshot
total_training_time = 0     # total training time
top_limit           = 98.5  # stop criteria validation accuracy threshold (got to argparse?)

no_criteria_flag = False

# print networks parameters on screen
helper.print_net_parameters(bs=args.bsize, vs=args.val_set, epochs=args.n_epochs, snap=args.snap, 
                            eval_criteria=args.eval_crit, use_criteria=use_criteria)

# creates a new accuracies file.csv
csv_filename = "%s_accuracies.txt" % training_id
helper.create_accuracy_csv_file(filename=csv_filename, traindir=args.train_dir, vs=args.val_set, 
                                height=HEIGHT, width=WIDTH, ch=CHANNELS, arch=args.arch, bs=args.bsize, 
                                epochs=args.n_epochs, ec=args.eval_crit, snap=args.snap, retraining=retraining)

# training operation: can stop by reaching the max number of iterations OR Ctrl+C OR by not evolving
if(retraining):
    # re-training operation. load trained model
    print("")
    print("[INFO] Loading pre-trained model for re-training...")  
    model.load(args.model_name)
    print("[INFO] Model:", args.model_name)
    print("[INFO] Trained model loaded!\n")

print("[INFO] Starting training operation...")
print("[INFO] Training ID:", training_id)
try:
    it = 0
    while(it < iterations):
        stime = time.time()
        train_acc, train_acc_nc = classifier.my_evaluate(model, X, Y, batch_size=128, criteria=args.eval_crit)
        val_acc, val_acc_nc = classifier.my_evaluate(model, Xv, Yv, batch_size=128, criteria=args.eval_crit)

        test_acc = 0
        test_acc_nc = 0
        min_acc = None
        if(args.test_dir is not None):
            test_acc, test_acc_nc = classifier.my_evaluate(model, Xt, Yt, batch_size=128, criteria=args.eval_crit)
        ftime = time.time() - stime

        # save best model if there is a better validation accuracy
        if(val_acc > best_val_acc or test_acc_nc > best_test_acc):
            no_progress = 0
            if val_acc > best_val_acc: best_val_acc = val_acc   
            if test_acc_nc > best_test_acc: best_test_acc = test_acc_nc
            
            # Save best model to file
            if(args.freeze):
                # NOTE: use it for freezing model
                del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
                model.save("./models/%s_frozen.tflearn" % training_id)

            print(colored("[INFO] Saving new best trained model soo far...", "yellow"))
            modelname = "./models/%s.tflearn" % training_id
            print(colored("[INFO] Model: %s" % modelname, "yellow"))
            model.save(modelname)
            print(colored("[INFO] Best trained model saved!\n", "yellow"))
            new_best = True
        else:
            no_progress += args.snap
            new_best = False

        # write to a .csv file the evaluation accuracy
        helper.write_accuracy_on_csv(filename=csv_filename, train_acc=(train_acc, train_acc_nc), 
                                     val_acc=(val_acc, val_acc_nc), test_acc=(test_acc, test_acc_nc), 
                                     time=total_training_time, best=new_best)
        
        # write accuracy's values on screen
        helper.print_accuracy(name="Evaluation", train_acc=train_acc, val_acc=val_acc, test_acc=test_acc, 
                              min_acc=min_acc, time=total_training_time, ctime=ftime)
        
        # NOTE: stop criteria check - accuracy AND no progress
        # NOTE/TODO: change to a callback
        if(use_criteria and helper.check_stop_criteria(val_acc=val_acc, maximum=99.5, no_progress=no_progress, limit=20*args.snap)): break
        if(use_criteria and False):
            # NOTE: for stop criteria experience
            if(val_acc_nc >= top_limit and not no_criteria_flag):
                print(colored("[INFO] NO_CRITERIA: Saving new best trained  model soo far...", "yellow"))
                modelname = "./models/nc_%s.tflearn" % training_id
                print(colored("[INFO] Model: %s" % modelname, "yellow"))
                model.save(modelname)
                print(colored("[INFO] NO_CRITERIA: Best trained model saved!\n", "yellow"))
                no_criteria_flag = True
            
            if(val_acc >= top_limit):
                print(colored("[INFO] W_CRITERIA: Saving new best trained  model soo far...", "yellow"))
                modelname = "./models/wc_%s.tflearn" % training_id
                print(colored("[INFO] Model: %s" % modelname, "yellow"))
                model.save(modelname)
                print(colored("[INFO] W_CRITERIA: Best trained model saved!\n", "yellow"))
                break

        # repeats the training operation until it reaches one stop criteria
        iteration_time = time.time()
        model.fit(X, Y, n_epoch=args.snap, shuffle=True, show_metric=True, batch_size=args.bsize, snapshot_step=False, 
                  snapshot_epoch=False, run_id=training_id, validation_set=(Xv, Yv), callbacks=None)
        iteration_time = time.time() - iteration_time
        total_training_time += iteration_time

        it += 1
# to stop the training at any moment by pressing Ctrl+C
except KeyboardInterrupt:
    # intermediate evaluation to check which is the best model once Ctrl+C was pressed
    train_acc, train_acc_nc = classifier.my_evaluate(model, X, Y, batch_size=128, criteria=args.eval_crit)
    val_acc, val_acc_nc = classifier.my_evaluate(model, Xv, Yv, batch_size=128, criteria=args.eval_crit)
    
    test_acc = 0
    test_acc_nc = 0
    min_acc = None
    if(args.test_dir is not None):
        test_acc, test_acc_nc = classifier.my_evaluate(model, Xt, Yt, batch_size=128, criteria=args.eval_crit)

# load best model (need this check if Ctr+C was pressed)
if(best_val_acc > val_acc or best_test_acc > test_acc_nc):
    print(colored("[INFO] Loading best trained model...", "yellow"))
    model.load("./models/%s.tflearn" % training_id)
    print(colored("[INFO] Model: models/%s.tflearn" % training_id, "yellow"))
    print(colored("[INFO] Restored the best model!", "yellow"))
else:
    # save actual trained model
    if(args.freeze):
        # NOTE: use it for freezing model
        del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
        model.save("./models/%s_frozen.tflearn" % training_id)

    print(colored("[INFO] Saving trained model...", "yellow"))
    modelname = "./models/%s.tflearn" % training_id
    print(colored("[INFO] Model: %s" % modelname, "yellow"))
    model.save(modelname)
    print(colored("[INFO] Trained model saved!\n", "yellow"))

# final evaluation with the best model
stime = time.time()
train_acc, train_acc_nc = classifier.my_evaluate(model, X, Y, batch_size=128, criteria=args.eval_crit)
val_acc, val_acc_nc = classifier.my_evaluate(model, Xv, Yv, batch_size=128, criteria=args.eval_crit)

test_acc = 0
test_acc_nc = 0
min_acc = None
if(args.test_dir is not None):
    test_acc, test_acc_nc = classifier.my_evaluate(model, Xt, Yt, batch_size=128, criteria=args.eval_crit)
ftime = time.time() - stime

# write to a .csv file the evaluation accuracy
helper.write_accuracy_on_csv(filename=csv_filename, train_acc=(train_acc, train_acc_nc), 
                             val_acc=(val_acc, val_acc_nc), test_acc=(test_acc, test_acc_nc), 
                             time=total_training_time, best=new_best)

# write accuracy's values on screen
helper.print_accuracy(name="Final Eval", train_acc=train_acc, val_acc=val_acc, test_acc=test_acc, 
                      min_acc=min_acc, time=None, ctime=ftime, color="green")

helper.print_memory_usage()

# NOTE: Turn show_image to FALSE when scheduling many trainings
classifier.test_model_accuracy(model=model, image_set=Xv, label_set=Yv, eval_criteria=args.eval_crit, 
                               show_image=args.show, cmatrix=None)

# sound a beep to notify that the training ended
print(colored("[INFO] Training complete!\a","green"))

# Report plot function
print("[INFO] Generating PDF refort plot...")
plot.parse_report_file(files_dir=csv_filename, title=training_id, xlabel="Epochs", ylabel="Accuracy (%)",
                       snap=args.snap, show=args.show)

print("[INFO] Report plot generated!")
print(colored("[INFO] All done!\a","green"))