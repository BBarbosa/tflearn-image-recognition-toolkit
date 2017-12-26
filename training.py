"""
Training script written in Tensorflow + TFLearn
For image classification
"""

from __future__ import division, print_function, absolute_import

import sys
import os
import platform
import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn
import tensorflow as tf

from tflearn.layers.core import input_data
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from utils import architectures, dataset, classifier, helper
from colorama import init
from termcolor import colored

# init colored print
init()

# argument parser
parser = argparse.ArgumentParser(description="High level Tensorflow + TFLearn training script.", 
                                 prefix_chars='-')
# required arguments
parser.add_argument("--data_dir", required=True, help="directory to the training data", type=str)
parser.add_argument("--arch", required=True, help="architecture name", type=str)
parser.add_argument("--run_id", required=True, help="model's path", type=str)
# optional arguments
parser.add_argument("--bsize", required=False, help="batch size (default=16)", default=16, type=int)
parser.add_argument("--test_dir", required=False, help="directory to the testing data", type=str)
parser.add_argument("--height", required=False, help="images height (default=64)", default=64, type=int)
parser.add_argument("--width", required=False, help="images width (default=64)", default=64, type=int)
parser.add_argument("--val_set", required=False, help="percentage of training data to validation (default=0.3)", default=0.3, type=float)
parser.add_argument("--gray", required=False, help="convert images to grayscale (default=False)", default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--freeze", required=False, help="freeze graph (not for retraining) (default=False)", default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--snap", required=False, help="evaluate training frequency (default=5)", default=5, type=int)

# parse arguments
args = parser.parse_args()

print(args, "\n")

# images properties
HEIGHT = args.height
WIDTH  = args.width

# load dataset and get image dimensions
if(args.val_set):
    CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv, _, _ = dataset.load_dataset_windows(args.data_dir, HEIGHT, WIDTH, shuffled=True, 
                                                                                        validation=args.val_set, mean=False, 
                                                                                        gray=args.gray, save_dd=False)
    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS
else:
    CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, _, _, _, _= dataset.load_dataset_windows(args.data_dir, HEIGHT, WIDTH, 
                                                                                     shuffled=True, save_dd=False)
    Xv = Yv = []
""" """

# to load CIFAR-10 dataset or MNIST
if(False):
    print("[INFO] Loading dataset (from directory)...")

    #CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv = dataset.load_cifar10_dataset(data_dir=args.data_dir)
    CLASSES, X, Y, HEIGHT, WIDTH, CHANNELS, Xv, Yv = dataset.load_mnist_dataset(data_dir=args.data_dir)

    classifier.HEIGHT   = HEIGHT
    classifier.WIDTH    = WIDTH
    classifier.IMAGE    = HEIGHT
    classifier.CHANNELS = CHANNELS

    print("[INFO] \t         Path:", args.data_dir)
    print("[INFO] \tShape (train):", X.shape, Y.shape)
    print("[INFO] \tShape   (val):", Xv.shape, Yv.shape)
    print("[INFO] Data loaded!\n")

# load test images
# NOTE: Use dataset.load_test_images or dataset.load_dataset_windows like on X, Y, Xv and Yv
if(args.test_dir is not None):
    _, Xt, Yt, _, _, _, _, _, _, _ = dataset.load_dataset_windows(args.test_dir, HEIGHT, WIDTH, shuffled=True, 
                                                                  mean=False, gray=args.gray, save_dd=True)

# Real-time data preprocessing (samplewise or featurewise)
img_prep = ImagePreprocessing()
img_prep.add_samplewise_zero_center()
img_prep.add_samplewise_stdnorm()      
#img_prep.add_zca_whitening()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=5.)

# computational resources definition (made changes on TFLearn's config.py)
tflearn.init_graph(num_cores=8, allow_growth=True)

# network definition
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS],    # shape=[None, IMAGE, IMAGE] for RNN
                     data_preprocessing=img_prep,              # NOTE: always check PP
                     data_augmentation=None)                   # NOTE: always check DA

# build network architecture
network, _ = architectures.build_network(args.arch, network, CLASSES)

# model definition
model = tflearn.DNN(network, checkpoint_path="./models/%s" % args.run_id, tensorboard_dir="./logs/", 
                    max_checkpoints=None, tensorboard_verbose=0, best_val_accuracy=0.95, 
                    best_checkpoint_path=None)

# training parameters
EPOCHS = 500                    # maximum number of epochs
SNAP = args.snap                # evaluates network progress at each SNAP epochs
iterations = EPOCHS // SNAP     # number of iterations (or evaluations)
use_criteria = True             # use stop criteria
eval_criteria = 0.80            # evaluation criteria (confidence)

best_val_acc = 0                # best validation accuracy
best_test_acc = 0               # best test accuracy
no_progress = 0                 # counter of how many snapshots without learning process
iteration_time = 0              # time between each snapshot
total_training_time = 0         # total training time

# print networks parameters on screen
helper.print_net_parameters(bs=args.bsize, vs=args.val_set, epochs=EPOCHS, snap=SNAP, 
                            eval_criteria=eval_criteria, use_criteria=use_criteria)

# creates a new accuracies file.csv
csv_filename = "%s_accuracies.txt" % args.run_id
helper.create_accuracy_csv_file(filename=csv_filename, testdir=args.test_dir, traindir=args.data_dir, vs=args.val_set, 
                                height=HEIGHT, width=WIDTH, arch=args.arch, bs=args.bsize, epochs=EPOCHS, ec=eval_criteria)

# training operation: can stop by reaching the max number of iterations OR Ctrl+C OR by not evolving
it = 0
try:
    while(it < iterations):
        stime = time.time()
        train_acc = classifier.my_evaluate(model, X, Y, batch_size=args.bsize, criteria=eval_criteria)
        val_acc = classifier.my_evaluate(model, Xv, Yv, batch_size=args.bsize, criteria=eval_criteria)
        test_acc = classifier.my_evaluate(model, Xv, Yv, batch_size=args.bsize, criteria=0.1)
        min_acc = None
        ftime = time.time() - stime

        # save best model if there is a better validation accuracy
        if(val_acc > best_val_acc or test_acc > best_test_acc):
            no_progress = 0
            if val_acc > best_val_acc: best_val_acc = val_acc   
            if test_acc > best_test_acc: best_test_acc = test_acc
            # Save best model to file
            if(args.freeze):
                # NOTE: use it for freezing model
                del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
                model.save("./models/%s_frozen.tflearn" % args.run_id)

            print(colored("[INFO] Saving new best trained model soo far...", "yellow"))
            modelname = "./models/%s.tflearn" % args.run_id
            print(colored("[INFO] Model: %s" % modelname, "yellow"))
            model.save(modelname)
            print(colored("[INFO] Best trained model saved!\n", "yellow"))
            new_best = True
        else:
            no_progress += 1
            new_best = False

        # write to a .csv file the evaluation accuracy
        helper.write_accuracy_on_csv(filename=csv_filename, train_acc=train_acc, val_acc=val_acc, test_acc=test_acc, 
                                     min_acc=min_acc, time=total_training_time, best=new_best)
        # write accuracy's values on file
        helper.print_accuracy(name="Evaluation", train_acc=train_acc, val_acc=val_acc, test_acc=test_acc, 
                              min_acc=min_acc, time=total_training_time, ctime=ftime)
        # NOTE: stop criteria check - accuracy AND no progress
        if(use_criteria and helper.check_stop_criteria(train_acc, val_acc, test_acc, 100, no_progress, 10)): break

        # repeats the training operation until it reaches one stop criteria
        iteration_time = time.time()
        model.fit(X, Y, n_epoch=SNAP, shuffle=True, show_metric=True, batch_size=args.bsize, snapshot_step=False, 
                  snapshot_epoch=False, run_id=args.run_id, validation_set=(Xv, Yv), callbacks=None)
        iteration_time = time.time() - iteration_time
        total_training_time += iteration_time

        it += 1
# to stop the training at any moment by pressing Ctrl+C
except KeyboardInterrupt:
    # intermediate evaluation to check which is the best model once Ctrl+C was pressed
    train_acc = classifier.my_evaluate(model, X, Y, batch_size=args.bsize, criteria=eval_criteria)
    val_acc = classifier.my_evaluate(model, Xv, Yv, batch_size=args.bsize, criteria=eval_criteria)
    test_acc = classifier.my_evaluate(model, Xv, Yv, batch_size=args.bsize, criteria=0.1)
    min_acc = None

# load best model (need this check if Ctr+C was pressed)
if(best_val_acc > val_acc or best_test_acc > test_acc):
    print(colored("[INFO] Loading best trained model...", "yellow"))
    model.load("./models/%s.tflearn" % args.run_id)
    print(colored("[INFO] Model: models/%s.tflearn" % args.run_id, "yellow"))
    print(colored("[INFO] Restored the best model!", "yellow"))
else:
    # save actual trained model
    if(args.freeze):
        # NOTE: use it for freezing model
        del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
        model.save("./models/%s_frozen.tflearn" % args.run_id)

    print(colored("[INFO] Saving trained model...", "yellow"))
    modelname = "./models/%s.tflearn" % args.run_id
    print(colored("[INFO] Model: %s" % modelname, "yellow"))
    model.save(modelname)
    print(colored("[INFO] Trained model saved!\n", "yellow"))

# final evaluation with the best model
stime = time.time()
train_acc = classifier.my_evaluate(model, X, Y, batch_size=args.bsize, criteria=eval_criteria)
val_acc = classifier.my_evaluate(model, Xv, Yv, batch_size=args.bsize, criteria=eval_criteria)
test_acc = classifier.my_evaluate(model, Xv, Yv, batch_size=args.bsize, criteria=0.1)
min_acc = None
ftime = time.time() - stime

# write to a .csv file the evaluation accuracy
helper.write_accuracy_on_csv(filename=csv_filename, train_acc=train_acc, val_acc=val_acc, 
                             test_acc=test_acc, min_acc=min_acc, time=total_training_time)
# write accuracy's values on file
helper.print_accuracy(name="Final Eval", train_acc=train_acc, val_acc=val_acc, test_acc=test_acc, 
                      min_acc=min_acc, time=None, ctime=ftime, color="green")
# NOTE: Turn show_image to False when scheduling many trainings
classifier.test_model_accuracy(model=model, image_set=Xv, label_set=Yv, eval_criteria=eval_criteria, 
                               show_image=False, cmatrix=args.run_id)

# sound a beep to notify that the training ended
print(colored("[INFO] Training complete!\a","green"))