"""
Set of auxiliary functions
"""

from colorama import init
from termcolor import colored

init()


# ///////////////////////////// print functions /////////////////////////////
# function to print accuracy's values
def print_accuracy(name="Accuracies",
                   train_acc=None,
                   val_acc=None,
                   test_acc=None,
                   min_acc=None,
                   time=None,
                   ctime=None,
                   color="yellow"):
    """
    Function to print accuracy's values.

    Params:
        `name` - print's header 
        `train_acc` - training accuracy
        `val_acc` - validation accuracy
        `test_acc` - test accuracy
        `min_acc` - minimal accuracy
        `time` - time 
        `ctime` - classification time
    """
    print(colored("\n-------- %s ---------" % name, color))

    if (train_acc is not None): print("     Train:", train_acc, "%")
    if (val_acc is not None): print("Validation:", val_acc, "%")
    if (test_acc is not None): print("      Test:", test_acc, "%")
    if (min_acc is not None): print("       Min:", min_acc, "%")
    if (time is not None): print("Total time: %.3f seconds" % time)

    print(colored("-----------------------------", color))
    if (ctime is not None):
        print(colored("Time: %.3f seconds\n" % ctime, color))


# function to print network's parameters
def print_net_parameters(bs=None,
                         vs=None,
                         epochs=None,
                         snap=None,
                         use_criteria=None,
                         eval_criteria=None):
    """
    Function to print network's parameters.

    Params:
        `bs` - batch size
        `vs` - validation set percentage
        `epochs` - limit number of epochs
        `snap` - snapshot model tick rate (in epochs)
        `eval_criteria` - evalutaion criteria to well classified image
    """
    print("-------------------")
    if (bs is not None): print("Batch size:", bs)
    if (vs is not None): print("Validation:", vs * 100, "%")
    if (epochs is not None): print("Max Epochs:", epochs)
    if (snap is not None): print("  Snapshot:", snap)
    if (use_criteria is not None): print(" Use crit.:", use_criteria)
    if (eval_criteria is not None): print("Eval crit.:", eval_criteria)
    print("-------------------\n")


# /////////////////////// functions to deal with .csv files //////////////////////
# function that creates a csv accuracy file
def create_accuracy_csv_file(filename="accuracy.txt",
                             traindir=None,
                             vs=None,
                             height=None,
                             width=None,
                             ch=None,
                             arch=None,
                             bs=None,
                             snap=None,
                             epochs=None,
                             ec=None,
                             retraining=False):
    """
    Function to create a accuracy .csv file.

    Params:
        `filename` - filename.csv 
    """

    if (retraining):
        return None

    fcsv = open(filename, "w+")

    fcsv.write("################ TRAINING REPORT #################\n")
    fcsv.write("# Images path   | %s\n" % traindir)
    fcsv.write("# Validation    | %.2f\n" % vs)
    fcsv.write("# Images shape  | (%d,%d,%d)\n" % (height, width, ch))
    fcsv.write("# Architecure   | %s\n" % arch)
    fcsv.write("# Batch size    | %d\n" % bs)
    fcsv.write("# Snap/Epoch    | %d\n" % snap)
    fcsv.write("# Max epochs    | %d\n" % epochs)
    fcsv.write("# Eval criteria | %.2f\n" % ec)
    fcsv.write("##################################################\n")
    fcsv.write("train,trainNC,val,valNC,test,testNC,time,new_best\n")

    fcsv.close()


# function to write accuracy on .csv file
def write_accuracy_on_csv(filename="accuracy.txt",
                          train_acc=None,
                          val_acc=None,
                          test_acc=None,
                          min_acc=None,
                          time=None,
                          best=False):
    """
    Function to write accuracy values on a .csv formated file.

    Params:
        `filename` - .csv filename
        `train_acc` - training accuracy
        `val_acc` - validation accuracy
        `test_acc` - test accuracy
        `min_acc` - minimum accuracy (not used temporarly)
        `time` - time
    """

    fcsv = open(filename, "a+")
    fcsv.write("%.2f,%.2f,%.2f,%.2f," % (train_acc[0], train_acc[1],
                                         val_acc[0], val_acc[1]))
    fcsv.write("%.2f,%.2f,%.3f,%s\n" % (test_acc[0], test_acc[1], time,
                                        str(best)))
    fcsv.close()


# function to write a string to a file
def write_string_on_file(filename="accuracy.txt", line="", first=False):
    """
    Function to write a string on a file

    Params:
        `filename` - name/path to file
        `line` - string to write on file
        `first` - enable/disabke first write 
    """

    if (first):
        fcsv = open(filename, "w+")
    else:
        fcsv = open(filename, "a+")

    fcsv.write(line)
    fcsv.close()


# ////////////////////////// other auxiliary functions /////////////////////////
# function to check stop criteria
def check_stop_criteria(val_acc,
                        no_progress,
                        limit,
                        train_acc=None,
                        test_acc=None,
                        maximum=97.5):
    """
    Function to check the stop criteria.

    Params:
        `train_acc` - training accuracy
        `val_acc` - validation accuracy
        `test_acc` - test accuracy
        `maximum` - maximum accuracy stop criteria
        `limit` - stop criteria for no progress 
    """

    return val_acc >= maximum or no_progress > limit


# function to measure the RAM usage
def print_memory_usage():
    """ 
    Code from
    http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/

        Return: Memory usage in MB
    """
    from memory_profiler import memory_usage
    import os

    mem = memory_usage(os.getpid())[0]
    print("[INFO] Memory usage %.2f MB" % mem)
    return mem