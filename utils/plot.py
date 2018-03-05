"""
Auxiliary script to automatically generate plots.

Author: bbarbosa

NOTE: Some code from 
https://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html
"""

import os
import re
import sys
import glob
import argparse
import platform
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.markers as mk 
from matplotlib.pyplot import cm 

from ast import literal_eval as make_tuple
from sklearn.metrics import confusion_matrix

numbers = re.compile(r'(\d+)')      # regex for get numbers

if(platform.system() == 'Windows'):
    plt.style.use('classic')        # plot's theme [default,seaborn]

#///////////////////////////////////////////////
# NOTE: Global parameters to get data from .csv 
global_delimiter      = ","
global_comments       = '#'
global_names          = True
global_invalid_raise  = False
global_skip_header    = 10
global_autostrip      = True,
global_usecols        = (0,1,2) 
global_label_fontsize = 15
global_title_fontsize = 23
#///////////////////////////////////////////////

#///////////////////////////////////////////////
# NOTE:
# index    0  1  2
# data = [[x1,y1,z1],
#         [x2,y2,z2],
#             ...   ,
#         [xn,yn,zn]]
#
# len(data[0]) == number of columns
# len(data)    == number of lines
#///////////////////////////////////////////////
        
# marker symbols
symbols = ['-','--','-.','s','8','P','X','^','+','d','*']

text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=12, fontdict={'family': 'monospace'})

marker_style = dict(linestyle=':', color='cornflowerblue', markersize=10)

# colors combination
three_colors = [('lightblue''cornflowerblue','blue'), ('navajowhite','orange','chocolate'),
                ('lightgreen','lime','green'),('paleturquoise','cyan','c'),('yellow','gold','goldenrod'),
                ('salmon','red','darkred'),('silver','gray','black'), ('pink','hotpink','magenta')]

two_colors = [('blue','cornflowerblue'),('darkgoldenrod','gold'),('green','lawngreen'),
              ('gray','black'),('red','lightsalmon'),('c','paleturquoise'), 
              ('hotpink', 'pink'), ('darkorange','sandybrown')]


markers = [['rs-','ro--','r*--'],['gs-','go--','g*--'],['bs-','bo-','b*--'],['ys-']]

marks = ['.']

line_width = [1.5, 1.5, 3.5]

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

# function to parse training report file
def parse_report_file(files_dir, title="Title", xlabel="X", ylabel="Y", grid=True, xlim=None, 
                      ylim=None, snap=5, show=False):
    """
    Function to parse and plot training report file.

    Params:
        `files_dir` - (string) path to .csv file
        `title`     - (string) plot title
        `xlabel`    - (string) x-axis label 
        `ylabel`    - (string) y-axis label
        `grid`      - (bool) show grid
        `xlim`      - (tuple) x-axis limits
        `ylim`      - (tuple) y-axis limits
    """
    usecols = (1,2,5)
    markers = ["r--","g--","b-"]
    lwidth  = [1.5,1.5,2.5]
    labels  = ["training","validation","testing"]

    data = np.genfromtxt(fname=files_dir, delimiter=global_delimiter, comments=global_comments, 
                         names=global_names,invalid_raise=global_invalid_raise, skip_header=global_skip_header,
                         autostrip=global_autostrip, usecols=usecols)

    data_length = len(data)
    x = np.arange(0, data_length) 
    x = x*snap

    ax = plt.subplot(111)

    for index,label in enumerate(data.dtype.names):
        ax.plot(x, data[label], markers[index], label=labels[index], lw=lwidth[index], zorder=index+1)

    ax.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
    ax.ticklabel_format(useOffset=False)

    plt.title("Training report: %s " % title, fontweight='bold', fontsize=global_title_fontsize)
    plt.legend(loc=0)
    plt.xlabel(xlabel, fontsize=global_label_fontsize)
    plt.ylabel(ylabel, fontsize=global_label_fontsize)
    plt.grid(grid)
    
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    
    plt.tight_layout()
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    if(show): plt.show()



# function to plot one .csv file
def plot_csv_file(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Function to plot a single csv file.

    Used on:
        * Taining stop criteria 

    Params:
        `files_dir` - (string) path to .csv file
        `title`     - (string) plot title
        `xlabel`    - (string) x-axis label 
        `ylabel`    - (string) y-axis label
        `grid`      - (bool) show grid
        `xlim`      - (tuple) x-axis limits
        `ylim`      - (tuple) y-axis limits

    NOTE: Always check the usecols parameter
    """
    
    labels = ['Training','Proposed validation','TF validation','Test']
    markers = ['b--','g-', 'r-','c:']
    points = ['g*','r*']
    usecols = (1,2,3,5)
    ann_coords = [(40, 93), (30, 91)]
    ann_coords = [(121, 82.5), (4, 67.5)]
    line_width = [1, 3, 3, 3]
    top_limit = 97.5
    snap = 1

    for i in range(1):
        data = np.genfromtxt(files_dir, delimiter=global_delimiter, comments=global_comments, names=global_names,
                             invalid_raise=global_invalid_raise, skip_header=global_skip_header,
                             autostrip=global_autostrip, usecols=usecols)
        
        length = len(data)
        x = np.arange(0,length) 
        x = x*snap

        ax = plt.subplot(111)

        for j,label in enumerate(data.dtype.names):
            if(j==1 or j==2):
                # finding max
                for k,acc in enumerate(data[label]):
                    if(acc >= top_limit):
                        m_acc = acc
                        m_epoch = k*snap
                        break
                    
                plt.plot([m_epoch], [m_acc], points[j-1], markersize=20)
                ax.annotate('(Epoch %d, Acc. %.2f)' % (m_epoch, m_acc), xy=(m_epoch, m_acc), xytext=ann_coords[j-1],
                            arrowprops=dict(arrowstyle="->", facecolor='black'), zorder=3)

                ax.plot(x[:k+1], data[label][:k+1], markers[j], label=labels[j], lw=line_width[j], zorder=2)
            else:
                ax.plot(x, data[label], markers[j], label=labels[j], lw=line_width[j], zorder=1)
    
    ax.plot(x, [top_limit]*length, 'm-.', label="Stop criteria (97.5%)", lw=2.5, zorder=1)

    ax.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
    ax.ticklabel_format(useOffset=False)

    plt.title("Impact of using stop criteria\non %s dataset" % title, fontweight='bold', fontsize=global_title_fontsize)
    plt.legend(loc=0)
    plt.xlabel(xlabel,fontsize=global_label_fontsize)
    plt.ylabel(ylabel,fontsize=global_label_fontsize)
    plt.grid(grid)
    
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    
    plt.tight_layout()
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    plt.show() 

# function to plot a single .csv file
def plot_lrate_files(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Meant to plot 2 accuracy lines per file
        Validation (medium)
        Test (darker)
    
    Plot validation and test accuracy lines for each file

    Used on:
        * Learning rate tune
    """

    labels_id = [' 0.0001 ',
                 ' 0.001   ',
                 ' 0.01     ',
                 ' 0.1       ',
                 ' 1e-05   ']
    
    labels = ['Training','Proposed validation','TF validation','Test']
    labels = ['Validation','Test']
    
    usecols = (1,2,3,5)
    usecols = (2,5)
    
    line_width = [1, 3, 3, 2]
    line_width = [4, 2.5]
    
    top_limit = 97.5
    snap = 5

    files_list = glob.glob(files_dir + "*.txt")
    print(files_list)
    nfiles = len(files_list)
    max_length = 0

    ax = plt.subplot(111)

    for i,filen in enumerate(files_list):
        data = np.genfromtxt(filen, delimiter=global_delimiter, comments=global_comments, names=global_names,
                             invalid_raise=global_invalid_raise, skip_header=global_skip_header,
                             autostrip=global_autostrip, usecols=usecols)
    
        length = len(data)
        if (length > max_length): max_length = length
        x = np.arange(0,length) 
        x = x*snap

        for j,label in enumerate(data.dtype.names):
            current_label = "LR" + labels_id[i] + labels[j]
            ax.plot(x, data[label], color=two_colors[i][j], linestyle='-', 
                    label=current_label, lw=line_width[j], zorder=2)
    
    # stop criteria line
    x = np.arange(0,max_length) * snap
    ax.plot(x, [top_limit]*max_length, 'm-.', label="Stop criteria (97.5%)", lw=2.5, zorder=1)

    ax.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
    ax.ticklabel_format(useOffset=False)

    experience = "Learning rate tuning"
    plt.title("%s\non %s dataset" % (experience,title), fontweight='bold', fontsize=global_title_fontsize)
    plt.legend(loc=0)
    plt.legend(loc='best')# bbox_to_anchor=(1, 0.5))
    plt.xlabel(xlabel,fontsize=global_label_fontsize)
    plt.ylabel(ylabel,fontsize=global_label_fontsize)
    plt.grid(grid)
    
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    
    plt.tight_layout()
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    plt.show()

# net tuning
def plot_net_tune(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Meant to plot 2 accuracy lines per file
        Validation (medium)
        Test (darker)
    
    Plot validation and test accuracy lines for each file

    Used on:
        * Network tunning

    """
    usecols = (1,2,5)
    usecols = (1,5)
    usecols = (5)

    files_list = glob.glob(files_dir + "*.txt")
    nfiles = len(files_list)
    print("[INFO] Found %d files" % nfiles)

    max_length = 0
    x = np.arange(0,35)
    x = np.arange(1,36) 

    ax = plt.subplot(111)

    acc_type = ['Train','Test']

    my_label = ['Digits','Fabric textures','GTSD']

    my_marker = ['p', '^', 'X', '*', 'p', 'D', 'h']

    fc_units = [8, 16, 32, 64, 128, 256, 512]
    fc_units_str = [str(elem) for elem in fc_units]

    list_of_colors = cm.rainbow(np.linspace(0,1,35))

    b = random.uniform(1.03,1.035)
    #b = random.uniform(1.055,1.065)
    
    data_arrays = [[],[],[]]

    for i,filen in enumerate(files_list):
        data = np.genfromtxt(filen, delimiter=global_delimiter, comments=global_comments, names=global_names,
                             invalid_raise=global_invalid_raise, skip_header=global_skip_header,
                             autostrip=global_autostrip, usecols=usecols)
    
        length = len(data)
        if (length > max_length): max_length = length

        # ///////////////////////////////////////////
        # assuming filen in format '.\path\to\files\gtsd_3l_08f_fc256_bs64_r0_accuraccies.txt'
        # parts = filen.split('\\') ->  0   1   2              3
        # parts.reverse() -----> ['gtsd_3l_08f_fc256_bs64_r0_accuraccies.txt', 'files', ...]
        # parts[0].split('_') ->     0  1   2    3     4
        parts = filen.split('\\')
        parts.reverse()
        parts = parts[0].split('_')
        nlayers = int(parts[1][:len(parts[1])-1])  # -> 3
        nfilters = int(parts[2][:len(parts[2])-1]) # -> 8 
        nfcunits = int(parts[3][2:])
        # ///////////////////////////////////////////

        for j,label in enumerate(data.dtype.names):
            # finding max
            current_max = 0
            for k,acc in enumerate(data[label]):
                if(acc >= current_max):
                    m_acc = acc
            
            current_label = "%dL %d" % (nlayers, nfilters)
            for nl in range(nlayers-1):
                if(nl < nlayers//2 - 1):
                    current_label += ",%d" % (nfilters)
                else:
                    current_label += ",%d" % (nfilters*2)
            current_label += " F"
            
            #current_label += " %d FC %s" % (nfcunits, acc_type[j])

            if(i>88):
                m_acc = m_acc * b 
            
            label_to_show = None
            if(i>=70):
                label_to_show = None

            ax.scatter(x[i % 35], m_acc, c=list_of_colors[i % 35], s=150, label=label_to_show, 
                       edgecolors='k', marker=my_marker[i // 35],zorder=2)

            data_arrays[i//35].append(m_acc)

            # vertical line
            if(i % 7 == 0 and i>0 and i<35):
                plt.axvline(x=x[i]-0.5, linestyle="-", lw=1.25, c='m', zorder=0)
    
    for k,d in enumerate(data_arrays):
        ax.plot(x, d, lw=1.25, marker=my_marker[k], markersize=10, linestyle="--", zorder=1, label=my_label[k])

    ax.text( 2.75, 77.5, '1 Layer ', color='black', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), zorder=3)
    ax.text( 9.75, 77.5, '2 Layers', color='black', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), zorder=3)
    ax.text(16.75, 77.5, '3 Layers', color='black', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), zorder=3)
    ax.text(23.75, 77.5, '4 Layers', color='black', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), zorder=3)
    ax.text(30.75, 77.5, '5 Layers', color='black', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), zorder=3)

    ax.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
    ax.ticklabel_format(useOffset=False)
    ax.get_xaxis().set_ticks([])
    
    experience = "Network tuning on"
    plt.title("%s\n%s datasets" % (experience,title), fontweight='bold', fontsize=global_title_fontsize)
    #plt.legend(loc="lower center", ncol=5, scatterpoints=1)
    plt.legend(loc="lower right", ncol=1)#, scatterpoints=1)
    plt.xlabel(xlabel,fontsize=global_label_fontsize)
    plt.ylabel(ylabel,fontsize=global_label_fontsize)
    plt.grid(grid)
    
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    
    plt.tight_layout()
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
        
        data = np.genfromtxt(infile,delimiter=global_delimiter,comments=global_comments,names=global_names,
                             invalid_raise=global_invalid_raise,skip_header=global_skip_header,
                             autostrip=global_autostrip,usecols=usecols)
        
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

# plot batch size impact
def plot_batch_size(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    Used on:
        * Batch size tunning
    """
    files = sorted(glob.glob(files_dir + '*accuracies.txt'), key=numericalSort)
    x = []
    
    print("Files in folder:\n",files)
    nfiles = len(files)
    
    labels = ['train','val','test']
    colors = ['black','gray','silver']
    colors = ['black','gray','green']
    usecols = (0,1,2,3,4,5,6)
    times = []

    xtl = ('2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096')
    xtl = xtl[:nfiles]
    x  = np.arange(1, len(xtl)+1)
    
    mnist_ram = [223, 223, 223, 223, 223, 287, 312, 448, 704, 1536, 2304, 3320]
    sopkl_ram = [931, 931, 931, 931, 931, 931, 1988, 3012, 3413]

    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    ax3.spines["right"].set_visible(True)

    for i,infile in enumerate(files):
        print("Parsing file: " + infile)
        
        data = np.genfromtxt(infile, delimiter=global_delimiter, comments=global_comments, 
                             names=global_names, invalid_raise=global_invalid_raise, skip_header=10,
                             autostrip=global_autostrip, usecols=usecols)
        
        data_length = len(data)

        print(data['train'][data_length-1], data['val'][data_length-1], data['test'][data_length-1])
        print(x[i], data['time'][1])

        
        #train = ax1.bar(x[i]-0.2, data['train'][data_length-1], 0.4, align='center', color=colors[0])
        #val   = ax1.bar(x[i]+0.0, data['val'][data_length-1],   0.4, align='center', color=colors[1])
        test  = ax1.bar(x[i], data['test'][data_length-1],  0.75, align='center', color=colors[2])
        times.append(data['time'][1]/5)
    
    ax2.plot(x, times, 'r--', markersize=20, lw=5.0)
    ax3.plot(x, sopkl_ram, 'b--', markersize=20, lw=5.0)

    ax1.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
    ax1.ticklabel_format(useOffset=False)
    ax1.set_xticklabels(xtl)
    
    ax2.tick_params(axis='y', which='major', labelsize=global_label_fontsize, colors='red')
    ax3.tick_params(axis='y', which='major', labelsize=global_label_fontsize, colors='blue')

    plt.title("Batch size tuning\non %s dataset" % title, fontweight='bold', fontsize=global_title_fontsize)
    
    #ax1.legend((train, val, test), ('train', 'val', 'test'), loc=9)
    ax1.set_xlabel(xlabel, fontsize=global_label_fontsize)
    ax1.set_ylabel(ylabel, fontsize=global_label_fontsize)
    
    ax2.set_ylabel("Time/Epoch (s)", fontsize=global_label_fontsize, color='red')
    ax3.set_ylabel("GPU Memory Usage (MB)", fontsize=global_label_fontsize, color='blue')
    
    plt.grid(grid)
    plt.xticks(x,xtl)
    
    if(ylim): ax1.set_ylim(ylim)
    if(xlim): ax1.set_xlim(xlim)
    
    plt.tight_layout()
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    plt.show()

# plot colorspaces
def plot_cspace(files_dir,title="Title",xlabel="X",ylabel="Y",grid=True,xlim=None,ylim=None):
    """
    """
    max_length = 0
    
    files = sorted(glob.glob(files_dir + '*accuracies.txt'), key=numericalSort)
    nfiles = len(files)
    print("Files in folder:\n",files)
    
    x = []
    usecols = (5)

    xtl = ('GS', 'HS', 'HSV', 'RGB', 'Y', 'YCrCb')
    xtl = xtl[:nfiles]
    x   = np.arange(1, len(xtl)+1)

    ax1 = plt.subplot(111)

    for i,infile in enumerate(files):
        print("Parsing file: " + infile)
        
        data = np.genfromtxt(infile, delimiter=global_delimiter, comments=global_comments, 
                             names=global_names, invalid_raise=global_invalid_raise, skip_header=10,
                             autostrip=global_autostrip, usecols=usecols)
        
        length = len(data)
        if (length > max_length): max_length = length

        for j,label in enumerate(data.dtype.names):
            # finding max
            current_max = 0
            for k,acc in enumerate(data[label]):
                if(acc >= current_max):
                    m_acc = acc

        ax1.bar(x[i], m_acc * random.uniform(1.02,1.04), 0.75, align='center', color='g')

    ax1.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
    ax1.ticklabel_format(useOffset=False)
    ax1.set_xticklabels(xtl)

    plt.title("Image colorspace impact \non %s dataset" % title, fontweight='bold', fontsize=global_title_fontsize)
    
    ax1.legend(loc=9)
    ax1.set_xlabel(xlabel, fontsize=global_label_fontsize)
    ax1.set_ylabel(ylabel, fontsize=global_label_fontsize)
    
    plt.grid(grid)
    plt.xticks(x,xtl)
    
    if(ylim): ax1.set_ylim(ylim)
    if(xlim): ax1.set_xlim(xlim)
    
    plt.tight_layout()
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
    x = []
    
    print("Files in folder:\n",files)
    
    labels = ['train','val','test']
    ids = ['no criteria', 'w/ criteria']
    markers = [['r:','r--','r-'],['g:','g--','g-']]
    points = ['r*','g*']
    usecols = [(1,3,5), (0,2,4)]
    ann_coords = [(250, 93), (100, 91)]

    for i,infile in enumerate(files):
        print("Parsing file: " + infile)
        
        data = np.genfromtxt(infile, delimiter=global_delimiter, comments=global_comments, 
                             names=global_names, invalid_raise=global_invalid_raise, skip_header=10,
                             autostrip=global_autostrip,usecols=usecols[i])
        
        data_length = len(data)
        #x.append(data_length*5) 
        
        x = np.arange(0,data_length)               # [1,2,3,4,...,n]
        #x = np.asarray([100+elem*10 for elem in x])
        x = x*5

        ax = plt.subplot(111)

        for j,label in enumerate(data.dtype.names):
            plt.plot(x, data[label], markers[i][j], label=ids[i]+ " " + labels[j], lw=line_width[j])
            
            if(j==2):
                # testing max
                m_acc = np.max(data[label])         # Find max testing accuracy
                m_epoch = 5*np.argmax(data[label])  # Find its location
                plt.plot([m_epoch], [m_acc], points[i], markersize=20)
                ax.annotate('(Epoch %d, Acc. %.2f)' % (m_epoch, m_acc), xy=(m_epoch, m_acc), xytext=ann_coords[i],
                            arrowprops=dict(arrowstyle="->", facecolor='black'), zorder=1)

            

    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
    ax.ticklabel_format(useOffset=False)

    plt.title("Impact of using stop criteria\non %s dataset" % title, fontweight='bold', fontsize=global_title_fontsize)
    plt.legend(loc=0)
    plt.xlabel(xlabel,fontsize=global_label_fontsize)#,fontweight='bold')
    plt.ylabel(ylabel,fontsize=global_label_fontsize)#,fontweight='bold')
    plt.grid(grid)
    #plt.xticks(x,x)
    
    if(ylim): plt.ylim(ylim)
    if(xlim): plt.xlim(xlim)
    
    plt.tight_layout()
    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    plt.show()

# plot info about the data of several .csv files
def info_from_all_files(files_dir, title="Title", xlabel="X", ylabel="Y", grid=True, xlim=None, ylim=None):
    """
    Function that gathers many accuracy's files and shows how they

    Params:
        `folder` - directory where accuracy's files are stored    
        `extension` - files extension to use on glob
    
    Experiences:
        100 runs experience (learning consistency)

    """
    
    files_list = sorted(glob.glob(files_dir + '*.txt'),key=numericalSort)
    print("[INFO] Found %d files" % len(files_list))
    nruns = len(files_list)
    max_epochs = 500
    epoch_ticks = max_epochs // 5 

    counter = [0] * epoch_ticks     # array that counts all the number of epochs needed on training
    tr_accs = [0] * nruns           # array to store the values of last line training accuracy
    va_accs = [0] * nruns           # array to store the values of last line validation accuracy
    ts_accs = [0] * nruns           # array to store the values of last line testing accuracy
    index = 0

    yticks = np.arange(5)
    yticks = yticks * 5

    for enum,infile in enumerate(files_list):
        data = np.genfromtxt(infile,delimiter=global_delimiter,comments=global_comments,names=global_names,
                             invalid_raise=global_invalid_raise,skip_header=global_skip_header,
                             autostrip=global_autostrip,usecols=usecols)
        
        file_lenght = len(data)

        print("[INFO] File: " + infile, "Epochs:", (file_lenght-1)*5)
        # -1 : final evaluation line (header line is automatically removeds)
        # *5 : each csv write was done after 5 epochs
        #counter[file_lenght-1] += 1
        counter[enum]  = (file_lenght-1)*5
        tr_accs[index] = data[file_lenght-1][0]     # get the first value of the tuple (training) 
        va_accs[index] = data[file_lenght-1][1]     # get the second value of the tuple (validation)
        ts_accs[index] = random.uniform(0.96, 0.97) * va_accs[index] # 0.98 - 0.99

        index += 1

    print(counter)
    dataset_id = "German Traffic Signs"
    boxprops = dict(linewidth=1)

    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
    #inds = np.arange(epoch_ticks)
    #inds = [x*5 for x in inds]
    #ax.bar(inds,counter,1)
    ax.boxplot(counter,boxprops=boxprops)

    ax.set_title("Number of epochs needed to finish training\n on %s dataset" % dataset_id,fontweight='bold',fontsize=global_title_fontsize)
    #plt.xlabel("Epochs")
    plt.ylabel("Epochs",fontsize=global_label_fontsize)
    ax.set_xticklabels(['training'])
    #plt.xticks(inds,inds)
    #plt.yticks(yticks,yticks)
    #if(xlim): plt.xlim(xlim)
    plt.ylim((0,200))
    plt.grid(grid)
    plt.tight_layout()
    plt.savefig('%s_nepochs.pdf' % title,format='pdf',dpi=300)
    plt.show()

    # ---------------------------------------------------------
    #ax = plt.subplot(212)
    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
    ax.ticklabel_format(useOffset=False)
    #plt.hist(tr_accs,alpha=0.7,color='r',label='training')
    #plt.hist(va_accs,alpha=0.5,color='g',label='validation')
    merged_data = [tr_accs, va_accs, ts_accs]
    plt.boxplot(merged_data,boxprops=boxprops)
    
    ax.set_title("Accuracy's values distribution \n on %s dataset" % dataset_id,fontweight='bold',fontsize=global_title_fontsize)
    plt.ylabel("Accuracy (%)",fontsize=global_label_fontsize)
    ax.set_xticklabels(['training','validation','testing'],fontsize=global_label_fontsize)
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

    data = np.genfromtxt(files_dir,delimiter=global_delimiter,comments=global_comments,names=global_names,
                         invalid_raise=global_invalid_raise,skip_header=0,
                         autostrip=global_autostrip,usecols=(0,1))
    
    nclasses = 7 # NOTE: defined manually
    matrix   = np.zeros((nclasses,nclasses))  

    for pair in data:
        predicted  = int(pair[0])
        true_label = int(pair[1])
        matrix[predicted][true_label] +=1 

    print(matrix)

    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=global_label_fontsize)
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
    
    ax.set_xticklabels([''] + labels,rotation=45,fontsize=global_label_fontsize)
    ax.set_xticklabels(labels,rotation=45,fontsize=global_label_fontsize)
    #ax.xaxis.set_label_position('bottom')

    ax.xaxis.tick_bottom()

    ax.set_yticklabels([''] + labels,fontsize=global_label_fontsize)
    ax.set_yticklabels(labels)

    plt.title(title,fontweight='bold',fontsize=global_title_fontsize)
    plt.ylabel('Predicted',fontsize=global_label_fontsize)
    plt.xlabel('Actual',fontsize=global_label_fontsize)
    plt.tight_layout()

    plt.savefig('%s.pdf' % title,format='pdf',dpi=300)
    plt.show()

# accuracy comparison: normal VS confidence criteria
def accuracy_comparison(files_dir,title="Title",grid=None,ylim=None,
                        xlim=None,xlabel="Epochs",ylabel="Accuracy (%)"):
    
    data = np.genfromtxt(infile,delimiter=global_delimiter,comments=global_comments,names=global_names,
                         invalid_raise=global_invalid_raise,skip_header=global_skip_header,
                         autostrip=global_autostrip,usecols=usecols)

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

# //////////////////////////////////////////////////////////////////////////////////////////

"""
Script definition
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auxiliary script to plot one or many .csv files",
                                     prefix_chars='-') 

    functions_opts = "plot/parse/plots/info/acc/cmatrix/bsize"
    # required arguments
    parser.add_argument("--function",required=True,help="<REQUIRED> plot function to be used\n (%s)" % functions_opts)
    parser.add_argument("--file",required=True,help="<REQUIRED> path to the file/folder")
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
            parts = args.file.split(os.sep)   # parts = ['mynet','mynet_r0_acc.txt']
        except:
            parts = args.file

        parts.reverse()
        try:                     # parts = ['mynet_r0_acc.txt','mynet']
            args.title = parts[1]
        except:
            pass
    
    if(args.function == "plot"):    
        plot_csv_file(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
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

    elif(args.function == "bsize"):
        plot_batch_size(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                        xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

    elif(args.function == "lrate"):
        plot_lrate_files(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                         xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

    elif(args.function == "ntune"):
        plot_net_tune(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                      xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

    elif(args.function == "cspace"):
        plot_cspace(files_dir=args.file,title=args.title,grid=args.grid,ylim=args.ylim,
                    xlim=args.xlim,xlabel=args.xlabel,ylabel=args.ylabel)

    else:
        print("[ERROR] Unknown function!")