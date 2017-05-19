# Script to crop many images at once
# It uses an auxiliary .csv file with the crop's info
# .csv file format
# filename,width,height,x1,y1,x2,y2,classid 

import argparse
import numpy as np
from PIL import Image

# arguments' parser
parser = argparse.ArgumentParser(description="Auxiliary script to crop many images",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("file",help="path to the .csv file with crop infos")
parser.add_argument("directory",help="directory where images are stored")

args = parser.parse_args()
print(args)

# NOTE: always check delimiter and dtype
data = np.genfromtxt(args.file,delimiter=";",comments='#',names=True, 
                             skip_header=0,autostrip=True,dtype=None)

# get the length of the .csv file
length = len(data) 

for i in range(length):
    filename,width,height,x1,y1,x2,y2,classid = data[:][i]
    filename = filename.decode("utf-8")
    
    # using relative path
    path = "%s%s" % (args.directory,filename)
    #print(filename,width,height,x1,y1,x2,y2,classid)
    img = Image.open(path)
    img = img.crop((x1,y1,x2,y2))
    
    #img.show()
    #imgc.show()
    img.save(path)