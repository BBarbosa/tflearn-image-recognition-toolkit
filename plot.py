# Auxiliary script to easily generate plots
import re,sys,argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ast import literal_eval as make_tuple

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
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    plt.savefig('%s.png' % title)
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
    
    #markers = ['ro','g^','bs','y+','c-','']

    for label in data.dtype.names:
        plt.plot(x,data[label],label=label)
    
    plt.title(title,fontweight='bold')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.xticks(x,xticks)
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    plt.savefig('%s.png' % title)
    plt.show()

def parse_csv_files(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Function to parse several csv files. For example, for N files it gathers all 
    the data, calculate a mean and plots it. 

    Params:
        files_dir - directory where error files are stored (at least 2 files)
    """
    
    means = []

    for infile in sorted(glob.glob(files_dir + '*accuracies.txt'), key=numericalSort):
        print("File: " + infile)
        
        data = np.genfromtxt(infile,delimiter=",",comments='#',names=True, 
                             skip_header=0,autostrip=True)
        
        mean = [0] * len(data[0])          # creates an empty array to store mean of each line

        # calculate mean of one X element
        for i,label in enumerate(data.dtype.names):
            mean[i] = np.mean(data[label])
        
        mean  = np.around(mean,2)
        means = np.vstack((means,mean)) if len(means)>0 else mean
    
    x = np.arange(1,len(means)+1)
    means = np.asarray(means)

    for i,label in enumerate(data.dtype.names):
        plt.plot(x,means[:,i],label=label)
    
    xticks = x

    plt.title(title,fontweight='bold')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.xticks(x,xticks)
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    plt.savefig('%s.png' % title)
    plt.show()

"""
Script definition
"""

def limits(s):
    """
    Function to parse plot's limits as they are represented as tuples
    in format "(lower_limit,upper_limit)"
    """
    try:
        return make_tuple(s)
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y")
        return None



# NOTE: arguments' parser
parser = argparse.ArgumentParser(description="Auxiliary script to plot one or many .csv files",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("function",help="plot function to be used (plot/parse)")
parser.add_argument("file",help="path to the file/folder")
# optional arguments
parser.add_argument("-t","--title",help="plot's title (string)")
parser.add_argument("-x","--xlabel",help="plot's x-axis label (string)")
parser.add_argument("-y","--ylabel",help="plot's y-axis label (string)")
parser.add_argument("-g","--grid",help="toggle plot's grid (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("-xl","--xlim",help="x-axis limits (tuple)",type=limits) # issue with negative values: change prefix_char
parser.add_argument("-yl","--ylim",help="y-axis limits (tuple)",type=limits) # issue with negative values: change prefix_char

args = parser.parse_args()

print(args)

if(args.title == None):
    # NOTE: if title isn't specified then uses filename as title
    # args.file = 'mynet\\mynet_r0_accuracies.txt'
    try:
        parts = args.file.split("\\")                               # parts = [mynet','mynet_r0_acc.txt']
    except:
        parts = args.file
    last_part_index = max(len(parts)-1,0)                             # lpi = 1
    new_title = parts[last_part_index].split(".")[0].split("_")[0]    # new_title = 'mynet_r0'
    args.title = new_title

if(args.function == "plot"):    
    plot_csv_file(infile=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                  xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)
elif (args.function == "parse"):
    parse_csv_files(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                    xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)
else:
    print("ERROR: Unknown function!")