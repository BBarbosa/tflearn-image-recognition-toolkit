# call example: 
# $ python random_crop.py filename.jpg
# ----------------------------- 
from PIL import Image
import sys
import os
import random

CROP = 64
NIMAGES = 16

fname = os.path.splitext(sys.argv[1])[0]   # /canvas1/canvas1-a-p001.png -> /canvas1/canvas1-a-p001
img = Image.open(sys.argv[1]) 
width,height = img.size                       

limit_w = width-CROP           # to avoid segmentation faults (left vertical half of the image for trainning)  
limit_h = height-CROP          # to avoid segmentation faults                           

#print "\t  Width: %4d  Height: %4d" % (width,height)
#print "\tLimit-W: %4d Limit-H: %4d" % (limit_w,limit_h)

# random crop
for i in range(0,NIMAGES): 
    h = random.randint(0,limit_h)
    w = random.randint(0,limit_w)
    img2 = img.crop((w,h,w+CROP,h+CROP))
    filename = "../training/%s_%d.png" % (fname, i)  
    img2.save(filename)