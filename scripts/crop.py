# call example: 
# $ python crop.py filename.jpg
# ----------------------------- 
from PIL import Image
from random import random
import sys,os

CROP    = 96        # for square images

CROPw   = 64       # for rectangular images
CROPh   = 64        # for rectangular images

NIMAGES = 324

fname = os.path.splitext(sys.argv[1])[0]   # /canvas1/canvas1-a-p001.png -> /canvas1/canvas1-a-p001~

# loads image
img = Image.open(sys.argv[1])

# get image dimensions 
width,height = img.size                       

limit_w = width-CROPw             # to avoid segmentation faults 
limit_h = height-CROPh            # to avoid segmentation faults      

# step for sliding window 
stepW = limit_w // 250
stepH = limit_h // 1               

stepW = max(1, stepW)
stepH = max(1,stepH)

# sliding window to crop image
h=0
i=0
while (h <= limit_h):
    w=0
    while (w <= limit_w):
        # ALWAYS check the place where the crops are going to be stored!!!
        #filename = "../../../cropped/defects/%s_%d.jpg" % (fname, i)
        filename = "train/bad/%s_%d.jpg" % (fname, i)
        img2 = img.crop((w,h,w+CROPw,h+CROPh))
        img2.save(filename)
        w = w + stepW
        i = i + 1
    h = h + stepH        