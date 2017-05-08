# Auxiliary script to easily generate plots
import re,sys,argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

numbers = re.compile(r'(\d+)')      # regex for get numbers

def numericalSort(value):
    """
    Splits out any digits in a filename, turns it into an actual 
    number, and returns the result for sorting. Like windows file
    explorer.
    
    Code get from
    http://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def plot_accuracies(x,y,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Function that creates a plot according to X and Y. Lengths must match.
    Params:
        x - array for x-axis [0,1,2]
        y - assuming that it is a matrix so it can automatically generate legend [[3,4,5], [6,7,8], [9,0,1]]
    """
    legend = np.arange(0,len(y[0])).astype('str')      # creates array ['0','1','2',...,'n']
    xticks = x

    plt.plot(x,y)
    plt.title(title,fontweight='bold')
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.xticks(x,xticks)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.savefig(title + '.png   ')
    plt.show()

def plot_csv_file(infile,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Function to plot a single csv file.

    Params:
        `infile` - (string) path to .csv file
        `title` - (string) plot title
        `xlabel` - (string) x-axis label 
        `ylabel` - (string) y-axis label
        `grid` - (bool) show grid
        `xlim` - (tuple) x-axis limits
        `ylim` - (tuple) y-axis limits
    """
    
    data = np.genfromtxt(infile,delimiter=",",comments='#',names=True, 
                             skip_header=0,autostrip=True)
    
    x = np.arange(0,len(data))               # [1,2,3,4,...,n]
    #x = np.asarray([100+elem*10 for elem in x])
    x = x*5
    
    xticks = [8,16,32,48,64,80,96,128,256,512]      # a-axis values
    xticks = x
    
    for i,label in enumerate(data.dtype.names):
        plt.plot(x,data[label],label=label)

    plt.title(title,fontweight='bold')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.xticks(x,xticks)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.savefig('snet.png')
    plt.show()
    
# NOTE: build a parser to make it generic
parser = argparse.ArgumentParser()
parser.add_argument("file",help="Path to the file")
parser.add_argument("-t","--title",help="Plot's title")

args = parser.parse_args()

print(args)

plot_csv_file(args.file,args.title,"Epochs","%",ylim=(0,101))