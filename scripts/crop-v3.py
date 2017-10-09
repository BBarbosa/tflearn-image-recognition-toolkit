# Script to crop many images at once
# It uses an auxiliary .csv file with the crop's info
# .csv file format
# filename,width,height,x1,y1,x2,y2,classid 

import argparse
import numpy as np
from PIL import Image
import os
import glob

# arguments' parser
parser = argparse.ArgumentParser(description="Auxiliary script to crop many images by a .csv file",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("directory",help="directory where training images are stored")

args = parser.parse_args()
print(args)

def crop_files_in_dir(directory):
    """
    Auxiliary function to crop images inside a class folder
    """
    infile = glob.glob("%s/%s" % (directory,'*.csv'))[0]
    #print("\t",infile)

    # NOTE: always check delimiter and dtype
    data = np.genfromtxt(infile,delimiter=";",comments='#',names=True, 
                         skip_header=0,autostrip=True,dtype=None)

    # get the length of the .csv file
    length = len(data) 
    
    for i in range(length):
        filename,width,height,x1,y1,x2,y2,classid = data[:][i]
        filename = filename.decode("utf-8")
        
        # using relative path
        path = "%s/%s" % (directory,filename)
        #print(filename,width,height,x1,y1,x2,y2,classid)
        img = Image.open(path)
        img = img.crop((y1,x1,y2,x2))
        
        #img.show()
        #input("Any key...")
        
        img.save(path)

"""
Script definition
"""
for roots,dirs,files in os.walk(args.directory):
    if not dirs:
        #print(roots)
        crop_files_in_dir(roots)