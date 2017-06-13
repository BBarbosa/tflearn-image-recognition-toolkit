from __future__ import division, print_function, absolute_import
import os,sys,platform
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from colorama import init
from termcolor import colored

# init colored print
init()

MODE = 'RGB' # RGB
loaded = True

# check number of arguments
if (len(sys.argv) < 2):
    print(colored("Call: $ python scripts\preprocessing.py {image_path} [image_path]","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))

img      = scipy.ndimage.imread(sys.argv[1], mode=MODE)   # mode='L', flatten=True -> grayscale
img_mean = img - scipy.ndimage.measurements.mean(img)     # confirmed. check data_utils.py on github
img_std  = img / np.std(img)                              # confirmed. check data_utils.py on github
mean_std = img_mean / np.std(img)

subplots = [221,222,223,224]

try:
    img2 = scipy.ndimage.imread(sys.argv[2], mode=MODE)
except:
    loaded = False
    subplots = [121,122]
    pass

plt.title("Image Preprocessing")

ax = plt.subplot(subplots[0])
plt.imshow(img)
ax.set_title("Original")

ax = plt.subplot(subplots[1])
plt.imshow(img_mean)
ax.set_title("Mean")

if(False):
        ax = plt.subplot(223)
        plt.imshow(img_std)
        ax.set_title("Std")

        ax = plt.subplot(224)
        plt.imshow(mean_std)
        ax.set_title("Mean & Std")
    

if(loaded):
    img2_mean = img2 - scipy.ndimage.measurements.mean(img2)
    
    ax = plt.subplot(subplots[2])
    plt.imshow(img2)
    ax.set_title("Original")
    
    ax = plt.subplot(subplots[3])
    plt.imshow(img2_mean)
    ax.set_title("Mean")
plt.show()