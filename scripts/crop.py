# call example: 
# $ python crop.py filename.jpg
# ----------------------------- 
from PIL import Image
import sys,os

CROP      = 128
NIMAGES   = 324

fname = os.path.splitext(sys.argv[1])[0]   # /canvas1/canvas1-a-p001.png -> /canvas1/canvas1-a-p001~

# loads image
img = Image.open(sys.argv[1])

# get image dimensions 
width,height = img.size                       

#limit_w = (width // 2) - CROP    # to avoid segmentation faults (left vertical half of the image for trainning)  
limit_w = width-CROP             # to avoid segmentation faults 
limit_h = height-CROP            # to avoid segmentation faults      

# step for sliding window (total_images = 18 * 18)
stepW = limit_w // 10
stepH = limit_h // 10                     

# sliding window to crop image
h=0
i=0
while (h < limit_h):
    w=0
    while (w < limit_w):
        # ALWAYS check the place where the crops are going to be stored!!!
        filename = "../side%d/%s_%d.bmp" % (CROP,fname, i)
        img2 = img.crop((w,h,w+CROP,h+CROP))
        img2.save(filename)
        w = w + stepW
        i = i + 1
    h = h + stepH        