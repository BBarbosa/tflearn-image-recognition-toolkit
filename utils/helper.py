#---------------------------------------
# Set of auxiliary printable functions
#---------------------------------------
from colorama import init
from termcolor import colored

init()

#-------------------------------------------------------------------------------
# print functions
#-------------------------------------------------------------------------------
# function to print accuracy's values
def print_accuracy(name="Accuracies",train_acc=None,val_acc=None,test_acc=None,
                   min_acc=None,time=None,ctime=None, color="yellow"):
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
    print(colored("\n-------- %s ---------" % name,color))
    
    if(train_acc is not None): print("     Train:", train_acc, "%")
    if(val_acc is not None):   print("Validation:", val_acc, "%")
    if(test_acc is not None):  print("      Test:", test_acc, "%")
    if(min_acc is not None):   print("       Min:", min_acc, "%")
    if(time is not None):      print("Total time: %.3f seconds" % time)
    
    print(colored("-----------------------------",color))
    if(ctime is not None): print(colored("Time: %.3f seconds\n" % ctime,color))

# function to print network's parameters
def print_net_parameters(bs=None,vs=None,epochs=None,snap=None,use_criteria=None,
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
    if(bs is not None):            print("Batch size:", bs)
    if(vs is not None):            print("Validation:", vs*100, "%")
    if(epochs is not None):        print("Max Epochs:", epochs)
    if(snap is not None):          print("  Snapshot:", snap)
    if(use_criteria is not None):  print(" Use crit.:", use_criteria)
    if(eval_criteria is not None): print("Eval crit.:", eval_criteria)
    print("-------------------\n")

#-------------------------------------------------------------------------------
# functions to deal with .csv files
#-------------------------------------------------------------------------------
# function that creates a csv accuracy file
def create_accuracy_csv_file(filename="accuracy.txt",testdir=None,traindir=None,vs=None,
                             height=None,width=None,arch=None,bs=None,epochs=None,ec=None):
    """
    Function to create a accuracy .csv file.

    Params:
        `filename` - filename.csv 
        `testdir` - test data directory (used as a boolean)
    """
    fcsv = open(filename,"w+")
                
    fcsv.write("################ TRAINING REPORT #################\n")
    fcsv.write("# Images path   | %s\n"   % traindir)
    fcsv.write("# Validation    | %.2f\n" % vs)
    fcsv.write("# Height        | %d\n"   % height)
    fcsv.write("# Width         | %d\n"   % width)
    fcsv.write("# Architecure   | %s\n"   % arch)
    fcsv.write("# Bacth Size    | %d\n"   % bs)
    fcsv.write("# Epochs        | %d\n"   % epochs)
    fcsv.write("# Eval Criteria | %.2f\n" % ec)
    fcsv.write("##################################################\n")
    
    if(testdir is not None):
        fcsv.write("train,validation,test,min,time\n")
    else:
        fcsv.write("train,validation,time\n")
    fcsv.close()

# function to write accuracy on .csv file
def write_accuracy_on_csv(filename="accuracy.txt",train_acc=None,val_acc=None,
                          test_acc=None,min_acc=None,time=None,best=False):
    """
    Function to write accuracy values on a .csv formated file.

    Params:
        `filename` - .csv filename
        `train_acc` - training accuracy
        `val_acc` - validation accuracy
        `test_acc` - test accuracy
        `min_acc` - minimum accuracy
        `time` - time
    """

    fcsv = open(filename,"a+")
    if(test_acc is not None):
        if(best):
            fcsv.write("%.2f,%.2f,%.2f,%.3f,best\n" % (train_acc,val_acc,test_acc,time))
        else:
            fcsv.write("%.2f,%.2f,%.2f,%.3f\n" % (train_acc,val_acc,test_acc,time))
    else:
        if(best):
            fcsv.write("%.2f,%.2f,%.2f,best\n" % (train_acc,val_acc,time))
        else:
            fcsv.write("%.2f,%.2f,%.2f\n" % (train_acc,val_acc,time))
    fcsv.close()

#-------------------------------------------------------------------------------
# other auxiliary functions
#-------------------------------------------------------------------------------
# function to check stop criteria
def check_stop_criteria(train_acc,val_acc,test_acc,maximum,no_progress,limit):
    """
    Function to check the stop criteria.

    Params:
        `train_acc` - training accuracy
        `val_acc` - validation accuracy
        `test_acc` - test accuracy
        `maximum` - maximum accuracy stop criteria
        `limit` - stop criteria for no progress 
    """
    if(test_acc is not None):
        if((train_acc > maximum and val_acc > maximum and test_acc > maximum) or no_progress >= limit):
            return True
    else:
        if((train_acc > maximum and val_acc > maximum) or no_progress > limit):
            return True
    
    return False