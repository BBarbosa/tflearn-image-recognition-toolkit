# Auxiliary script to easily generate plots
import re,sys,argparse,platform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ast import literal_eval as make_tuple
import glob

numbers = re.compile(r'(\d+)')      # regex for get numbers

if(platform.system() == 'Windows'):
    plt.style.use('default')            # plot's theme [default,seaborn]

# function to sort string as the windows explorer does
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

# function that plots data stored as a matrix
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

# function to plot one .csv file
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
    
    length = len(data)
    x = np.arange(0,length)               # [1,2,3,4,...,n]
    #x = np.asarray([100+elem*10 for elem in x])
    x = x*10
    
    xticks = [8,16,32,48,64,80,96,128,256,512]      # a-axis values
    xticks = x

    yticks = [0,10,20,30,40,50,60,70,80,90,100]
    
    criteria = [97.5] * length
    #markers = ['ro','g^','bs','y+','c-','']

    plt.style.use('default')
    
    plt.plot(x,criteria,'r--',label="stop_criteria")
    for label in data.dtype.names:
        plt.plot(x,data[label],label=label)
    
    plt.title(title,fontweight='bold')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.xticks(x,xticks)
    plt.yticks(yticks)
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    plt.savefig('%s.png' % title)
    plt.show()

# function to parse several .csv files and join all the info (mean)
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

# plot several .csv files into one single plot
def plot_several_csv_files(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Function to plot several csv files at one shot. For example, for N files it gathers all 
    the data, calculate a mean and plots it. 

    Params:
        files_dir - directory where error files are stored (at least 2 files)
    """

    symbols = ['-','--',':','^']
    colors  = ['r','g','b','y']
    files = sorted(glob.glob(files_dir + '*accuracies.txt'), key=numericalSort)

    for infile,color,_ in zip(files,colors,[1]):
        print("Parsing file: " + infile)
        
        # len(data) == number of lines
        data = np.genfromtxt(infile,delimiter=",",comments='#',names=True, 
                             skip_header=0,autostrip=True)
        # data = [(x1,y1,z1),
        #         (x2,y2,z2),
        #             ...    ]
        
        # len(data[0]) == number of columns
        x = np.arange(0,len(data[0])) 

        ax = plt.subplot(111)
        ax.bar(x, data[data.dtype.names[0]], width=0.1,color='b',align='center')
        ax.bar(x, data[data.dtype.names[1]], width=0.1,color='g',align='center')
        ax.bar(x, data[data.dtype.names[2]], width=0.1,color='r',align='center')
        ax.bar(x, data[data.dtype.names[3]], width=0.1,color='y',align='center')

        #for label,symbol in zip(data.dtype.names,symbols):
        #    dot = '%s%s' % (color,symbol)
        #    #plt.plot(x,data[label],dot,label=label)
        #    plt.barh(x,data[label],align='center',alpha=0.5)
    
    xticks = x*5

    plt.title(title,fontweight='bold')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.xticks(x,xticks)
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig('%s.png' % title)
    plt.show()

# plot info about the data of several .csv files
def info_from_all_files(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Function that gathers many accuracy's files and shows how they
    Used on 100 runs experience

    Params:
        folder - directory where accuracy's files are stored    
        extension - files extension to use on glob
    """
    
    nruns = 100
    max_epochs = 200
    epoch_ticks = max_epochs // 5 + 1

    counter = [0] * epoch_ticks     # array that counts all the number of epochs needed on training
    tr_accs = [0] * nruns           # array to store the values of last line training accuracy
    va_accs = [0] * nruns           # array to store the values of last line validation accuracy
    index = 0

    for infile in sorted(glob.glob(files_dir + '*accuracies.txt'),key=numericalSort):
        data = np.genfromtxt(infile,delimiter=",",comments='#',names=True, 
                             skip_header=0,autostrip=True)
        
        file_lenght = len(data)
        print("File: " + infile, "Epochs:", (file_lenght-1)*5)
        # -1 : final evaluation line (header line is automatically removeds)
        # *5 : each csv write was done after 5 epochs
        counter[file_lenght-1] += 1
        tr_accs[index] = data[file_lenght-1][0]     # get the first value of the tuple  
        va_accs[index] = data[file_lenght-1][1]     # get the second value of the tuple

        index += 1

    ax = plt.subplot(211)
    inds = np.arange(epoch_ticks)
    inds = [x*5 for x in inds]
    ax.bar(inds,counter,1)
    ax.set_title("Number of epochs needed to finish training",fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Counter")
    plt.xticks(inds,inds)
    plt.xlim((40,140))
    plt.grid(grid)
    plt.tight_layout()

    ax = plt.subplot(212)
    plt.hist(tr_accs,alpha=0.7,color='r',label='training')
    plt.hist(va_accs,alpha=0.5,color='g',label='validation')
    ax.set_title("Accuracy's values distribution",fontweight='bold')
    plt.xlabel("Accuracy")
    plt.ylabel("Counter")
    plt.grid(grid)
    plt.tight_layout()
    plt.legend()

    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    if(title): plt.savefig('%s.png' % title)
    plt.show()
 

# plot limits parser function
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

"""
Script definition
"""
# NOTE: arguments' parser
parser = argparse.ArgumentParser(description="Auxiliary script to plot one or many .csv files",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("function",help="plot function to be used (plot/parse/info)")
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
        parts = args.file.split("\\")                   # parts = ['mynet','mynet_r0_acc.txt']
    except:
        parts = args.file

    last_part_index = max(len(parts)-1,0)               # lpi = 1
    new_title = parts[last_part_index].split(".")[0]    # new_title = 'mynet_r0_acc'
    args.title = new_title

if(args.function == "plot"):    
    plot_csv_file(infile=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                  xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

elif(args.function == "parse"):
    plot_several_csv_files(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                    xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

elif(args.function == "info"):
    info_from_all_files(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                    xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)
else:
    print("ERROR: Unknown function!")