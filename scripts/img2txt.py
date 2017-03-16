import scipy.ndimage, sys
import numpy as np
from PIL import Image

if(len(sys.argv) < 3):
    print("Call: $ python img2txt.py {class} {image}")
    sys.exit("ERROR: Not enough arguments!")
else:
    # get command line arguments
    classid  = sys.argv[1]
    filename = sys.argv[2]
    
    background = Image.open(filename)
    width,height = background.size
    
    img = scipy.ndimage.imread(filename, flatten=True, mode='L')

    print classid,
    # goes throug the image
    for i in range(0,height):
        for j in range(0,width):
            print("%d" % img[i][j]),
    