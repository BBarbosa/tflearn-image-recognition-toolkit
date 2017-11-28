"""
Auxiliary script to automatically generate plots.

NOTE: Some code from 
https://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html
"""

import re,sys,argparse,platform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ast import literal_eval as make_tuple
import glob
from sklearn.metrics import confusion_matrix

numbers = re.compile(r'(\d+)')      # regex for get numbers

if(platform.system() == 'Windows'):
    plt.style.use('classic')        # plot's theme [default,seaborn]

########################################
# NOTE: Parameters to get data from .csv 
delimiter      = ","
comments       = '#'
names          = True
invalid_raise  = False
skip_header    = 10
autostrip      = True,
usecols        = (0,1,2) 
label_fontsize = 15
title_fontsize = 23
########################################

# NOTE
# index    0  1  2
# data = [(x1,y1,z1),
#         (x2,y2,z2),
#             ...    ]

# NOTE
# len(data[0]) == number of columns
# len(data)    == number of lines
        
# marker symbols
symbols = ['-','--','-.','s','8','P','X','^','+','d','*']

text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=12, fontdict={'family': 'monospace'})

marker_style = dict(linestyle=':', color='cornflowerblue', markersize=10)

# colors combination
colors = [('cornflowerblue','blue'),('navajowhite','orange'),('pink','hotpink'),('lightgreen','green'),
          ('paleturquoise','c'),('gold','goldenrod'),('salmon','red'),('silver','gray')]

markers = [['rs-','ro--','r*--'],['gs-','go--','g*--'],['bs-','bo-','b*--'],['ys-']]

line_width = [1.0,2.0,2.0]

# files ids
ids = ['1_','2_','4_','8_','16_','32_','64_','128_']

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
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    plt.show()

# function to plot one .csv file
def plot_csv_file(infile,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Function to plot a single csv file.

    Used on:
        accuracy comparison 

    Params:
        `infile` - (string) path to .csv file
        `title` - (string) plot title
        `xlabel` - (string) x-axis label 
        `ylabel` - (string) y-axis label
        `grid` - (bool) show grid
        `xlim` - (tuple) x-axis limits
        `ylim` - (tuple) y-axis limits

    NOTE: Always check the usecols parameter
    """
    
    data = np.genfromtxt(infile,delimiter=delimiter,comments=comments,names=names,
                         invalid_raise=invalid_raise,skip_header=skip_header,
                         autostrip=autostrip,usecols=usecols)
    
    length = len(data)
    x = np.arange(0,length)               # [1,2,3,4,...,n]
    #x = np.asarray([100+elem*10 for elem in x])
    x = x*5

    xticks = [8,16,32,48,64,80,96,128]      # a-axis values
    xticks = list(np.arange(0,10+1,5)) + list(np.arange(40,max(x)+1,30))
    xticks = np.arange(min(x), max(x)+1, 50)
    xticks = x

    print("xticks",xticks)
    
    yticks = [0,10,20,30,40,50,60,70,80,90,100]
    yticks = np.arange(0, 100+1, 20) 
    
    #criteria = [97.5] * length
    #plt.style.use('default')
    
    labels = ['training (confidence >=75%)','validation (confidence >=75%)','validation']
    markers = ['r-','r--','b--']

    #plt.plot(x,criteria,'r--',label="stop_criteria")
    for i,label in enumerate(data.dtype.names):
        plt.plot(x,data[label],markers[i],label=labels[i],markersize=1,linewidth=1)
    
    #plt.plot(x,[99]*length,'g-.',label="99% stop criteria",linewidth=1) # stop criteria plot
    
    plt.title(title,fontweight='bold')
    plt.title("Parking lot classification with a \n32 layer Residual Neural Network",fontweight='bold')
    plt.legend(loc=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.xticks(xticks,xticks)
    plt.yticks(yticks,yticks)
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
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
        
        data = np.genfromtxt(infile,delimiter=delimiter,comments=comments,names=names,
                             invalid_raise=invalid_raise,skip_header=skip_header,
                             autostrip=autostrip,usecols=usecols)
        
        mean = [0] * len(data[0])   # creates an empty array to store mean of each line
        print("mean",len(mean))

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
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    plt.show()

# plot several .csv files into one single plot
def plot_several_csv_files(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Function to plot several csv files at one shot. For example, for N files it gathers all 
    the data, calculate a mean and plots it. 

    Params:
        files_dir - directory where error files are stored (at least 2 files)

    Experiences:
        with vs without stop critera 
    """

    files = sorted(glob.glob(files_dir + '*accuracies.txt'), key=numericalSort)
    bar_width = 2
    x = []
    
    print("Files in folder:\n",files)
    labels = ['training (confidence >=75%)','validation (confidece >=75%)','validation']
    ids = ['no criteria', 'w/ criteria']

    markers = [['r-','r-','r:'],['g-','g-','g:']]

    i=0
    for infile,color,rid in zip(files,colors,ids):
        print("Parsing file: " + infile)
        
        data = np.genfromtxt(infile,delimiter=delimiter,comments=comments,names=names,
                             invalid_raise=invalid_raise,skip_header=skip_header,
                             autostrip=autostrip,usecols=(0,1,2))
        
        data_length = len(data)
        #x.append(data_length*5) 
        
        x = np.arange(0,data_length)               # [1,2,3,4,...,n]
        #x = np.asarray([100+elem*10 for elem in x])
        x = x*5

        ax = plt.subplot(111)

        for j,label in enumerate(data.dtype.names):
            # markevery=5 markers[j][i]
            plt.plot(x,data[label],markers[i][j],label=ids[i]+ " " + labels[j],lw=line_width[j])

        #plt.plot(x,data[0],symbols[0],label=labels[0]) # training
        #plt.plot(x,data[1],symbols[1],label=labels[1]) # validation >80
        #plt.plot(x,data[2],symbols[2],label=labels[2]) # validation
        
        # bar chart
        #ax.bar(data_length*5-bar_width/2, data[data.dtype.names[0]][data_length-1], 
        #       width=bar_width,color=color[0],align='center',label=rid+data.dtype.names[0]) # train_acc
        #
        #ax.bar(data_length*5+bar_width/2, data[data.dtype.names[1]][data_length-1], 
        #       width=bar_width,color=color[1],align='center',label=rid+data.dtype.names[1]) # test_acc

        i += 1

    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
    ax.ticklabel_format(useOffset=False)

    plt.title("Impact of using stop criteria\non %s dataset" % title,fontweight='bold',fontsize=title_fontsize)
    plt.legend(loc=0)
    plt.xlabel(xlabel,fontsize=label_fontsize)#,fontweight='bold')
    plt.ylabel(ylabel,fontsize=label_fontsize)#,fontweight='bold')
    plt.grid(grid)
    #plt.xticks(x,x)
    
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    
    plt.tight_layout()
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    plt.show()

# plot info about the data of several .csv files
def info_from_all_files(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Function that gathers many accuracy's files and shows how they

    Params:
        `folder` - directory where accuracy's files are stored    
        `extension` - files extension to use on glob
    
    Experiences:
        100 runs experience

    """
    
    files_list = sorted(glob.glob(files_dir + '*.txt'),key=numericalSort)
    print("[INFO] Found %d files" % len(files_list))
    nruns = len(files_list)
    max_epochs = 500
    epoch_ticks = max_epochs // 5 + 1

    counter = [0] * epoch_ticks     # array that counts all the number of epochs needed on training
    tr_accs = [0] * nruns           # array to store the values of last line training accuracy
    va_accs = [0] * nruns           # array to store the values of last line validation accuracy
    index = 0

    yticks = np.arange(5)
    yticks = yticks * 5

    for enum,infile in enumerate(files_list):
        data = np.genfromtxt(infile,delimiter=delimiter,comments=comments,names=names,
                             invalid_raise=invalid_raise,skip_header=skip_header,
                             autostrip=autostrip,usecols=usecols)
        
        file_lenght = len(data)

        print("[INFO] File: " + infile, "Epochs:", (file_lenght-1)*5)
        # -1 : final evaluation line (header line is automatically removeds)
        # *5 : each csv write was done after 5 epochs
        #counter[file_lenght-1] += 1
        counter[enum]  = (file_lenght-1)*5
        tr_accs[index] = data[file_lenght-1][0]     # get the first value of the tuple (training) 
        va_accs[index] = data[file_lenght-1][1]     # get the second value of the tuple (validation)

        index += 1

    dataset_id = "Fabric"
    boxprops = dict(linewidth=1)

    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
    #inds = np.arange(epoch_ticks)
    #inds = [x*5 for x in inds]
    #ax.bar(inds,counter,1)
    ax.boxplot(counter,boxprops=boxprops)

    ax.set_title("Number of epochs needed to finish \n training on %s dataset" % dataset_id,fontweight='bold',fontsize=title_fontsize)
    #plt.xlabel("Epochs")
    plt.ylabel("Epochs",fontsize=label_fontsize)
    ax.set_xticklabels(['training'])
    #plt.xticks(inds,inds)
    #plt.yticks(yticks,yticks)
    #if(xlim): plt.xlim(xlim)
    plt.grid(grid)
    plt.tight_layout()
    plt.savefig('%s_nepochs.pdf' % title,format='pdf',dpi=300)
    plt.show()

    # ---------------------------------------------------------
    #ax = plt.subplot(212)
    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
    ax.ticklabel_format(useOffset=False)
    #plt.hist(tr_accs,alpha=0.7,color='r',label='training')
    #plt.hist(va_accs,alpha=0.5,color='g',label='validation')
    merged_data = [tr_accs,va_accs]
    plt.boxplot(merged_data,boxprops=boxprops)
    
    ax.set_title("Accuracy's values distribution \n on %s dataset" % dataset_id,fontweight='bold',fontsize=title_fontsize)
    plt.ylabel("Accuracy (%)",fontsize=label_fontsize)
    ax.set_xticklabels(['training','validation'],fontsize=label_fontsize)
    plt.grid(grid)
    plt.tight_layout()
    plt.ylim(ylim)

    plt.savefig('%s_accuracy_distribution.pdf' % title,format='pdf',dpi=300)
    plt.show()

# generate confusion matrix    
def generate_cmatrix(files_dir,title="Title"):
    """
    Generate confusion matrix from 2 column file with (predicted,ground) format.abs

    Params:
        `files_dir` - path to confusion matrix file
        `title` - plot title 
    """

    data = np.genfromtxt(files_dir,delimiter=delimiter,comments=comments,names=names,
                         invalid_raise=invalid_raise,skip_header=0,
                         autostrip=autostrip,usecols=(0,1))
    
    nclasses = 7 # NOTE: defined manually
    matrix   = np.zeros((nclasses,nclasses))  

    for pair in data:
        predicted  = int(pair[0])
        true_label = int(pair[1])
        matrix[predicted][true_label] +=1 

    print(matrix)

    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
    #cax = ax.matshow(matrix)
    cax = ax.imshow(matrix,cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(cax)

    # add values to matrix
    for x in range(nclasses):
        for y in range(nclasses):
            ax.annotate(str(int(matrix[x][y])), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    labels = [str(elem+1) for elem in np.arange(nclasses)]
    #labels = ["canvas","cushion","linseeds","sand","seat","stone"] # NOTE: defined manually
    labels = ['0','1','2','3','4','5','6','7','8','9','10']
    labels = ['1','2','3','4','5','6','7']
    
    ticks = [int(elem)-1 for elem in labels]
    #ticks = [0,1,2,3,4,5,6,7,8,9,10]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    #print(ticks)
    
    ax.set_xticklabels([''] + labels,rotation=45,fontsize=label_fontsize)
    ax.set_xticklabels(labels,rotation=45,fontsize=label_fontsize)
    #ax.xaxis.set_label_position('bottom')

    ax.xaxis.tick_bottom()

    ax.set_yticklabels([''] + labels,fontsize=label_fontsize)
    ax.set_yticklabels(labels)

    plt.title(title,fontweight='bold',fontsize=title_fontsize)
    plt.ylabel('Predicted',fontsize=label_fontsize)
    plt.xlabel('Actual',fontsize=label_fontsize)
    plt.tight_layout()

    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    plt.show()

# accuracy comparison: normal VS confidence criteria
def accuracy_comparison(files_dir,title="Title",grid=None,ylim=None,
                        xlim=None,xlabel="Epochs",ylabel="Accuracy (%)"):
    
    data = np.genfromtxt(infile,delimiter=delimiter,comments=comments,names=names,
                         invalid_raise=invalid_raise,skip_header=skip_header,
                         autostrip=autostrip,usecols=usecols)

    file_lenght = len(data) - 1
    for i in range(file_lenght):
        plt.plot()
        plt.plot()
        plt.plot()

    return None

# funtion to plot data distribution
# TODO: add method to load data from files and extract counts
def plot_data_distribution(counts):
    """
    Function to plot data distribution.

    Params:
        `indices` - 
        `counts` - 
    """
    #print(colored("INFO: Saving data distribution image.","yellow"))
    indices = np.arange(len(counts))
    rects = plt.bar(indices,counts)
    plt.xlabel("Class")
    plt.xticks(indices)
    plt.ylabel("Count")
    plt.grid(True)
    image_title = train_path.split("\\")
    image_title.reverse()
    image_title = image_title[1]
    plt.title("Images distribution per class (%s)" % train_path,fontweight='bold')

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
    
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)

# plot limits parser function
def limits(s):
    """
    Function to parse plot's limits as they are represented as tuples
    in format "(lower_limit,upper_limit)"
    """
    try:
        return make_tuple(s)
    except:
        raise argparse.ArgumentTypeError("Coordinates must be formatted like (x,y)")
        return None

"""
Script definition
"""
# NOTE: arguments' parser
parser = argparse.ArgumentParser(description="Auxiliary script to plot one or many .csv files",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("--function",required=True,help="plot function to be used (plot/parse/plots/info/acc/cmatrix)")
parser.add_argument("--file",required=True,help="path to the file/folder")
# optional arguments
parser.add_argument("--title",help="plot's title (string)")
parser.add_argument("--xlabel",help="plot's x-axis label (string)")
parser.add_argument("--ylabel",help="plot's y-axis label (string)")
parser.add_argument("--grid",help="toggle plot's grid (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--xlim",help="x-axis limits (tuple)",type=limits) # issue with negative values: change prefix_char
parser.add_argument("--ylim",help="y-axis limits (tuple)",type=limits) # issue with negative values: change prefix_char

args = parser.parse_args()

print(args)

if(args.title == None):
    # NOTE: if title isn't specified then uses filename as title
    # args.file = 'mynet\\mynet_r0_accuracies.txt' OR
    # args.file = 'mynet\\mynet_folder\\'
    try:
        parts = args.file.split("\\")   # parts = ['mynet','mynet_r0_acc.txt']
    except:
        parts = args.file

    parts.reverse()
    try:                     # parts = ['mynet_r0_acc.txt','mynet']
        args.title = parts[1]
    except:
        pass
    
if(args.function == "plot"):
    #args.title = parts[0].split(".")[0]   # new_title = 'mynet_r0_acc'    
    plot_csv_file(infile=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                  xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

elif(args.function == "parse"):
    parse_csv_files(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                    xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

elif(args.function == "plots"):
    plot_several_csv_files(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                           xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

elif(args.function == "info"):
    info_from_all_files(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                        xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

elif(args.function == "acc"):
    accuracy_comparison(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                        xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

elif(args.function == "cmatrix"):
    generate_cmatrix(files_dir=args.file,title=args.title)


else:
    print("ERROR: Unknown function!")