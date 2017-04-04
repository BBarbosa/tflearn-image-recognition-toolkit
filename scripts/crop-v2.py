# call example: 
# $ python crop.py filename.jpg
# -----------------------------
# 
# TODO: check if crops fit in image (img.size)

from PIL import Image
import sys,os

if(len(sys.argv) < 2):
    sys.exit("ERROR! Not enough arguments!")

# loads image
fname = sys.argv[1].split(".")[0]
img   = Image.open(sys.argv[1])
width,height = img.size

print("File: ", sys.argv[1])
print("Dims: ",width, height)

CROP   = 96
NCROPS = 50 

xi = 0
yi = 0

xf = 96 
yf = 96
 
# check if selected area fits on image
if(xi > width or xf > width or
   xi < 0 or xf < 0 or 
   yi > height or yf > height or
   yi < 0 or yf < 0):
    sys.exit("ERROR! Selected area gets out of image's dimensions!")

xf -= CROP     # to ensure it doesn't get out of the image
yf -= CROP     # to ensure it doesn't get out of the image

# check that selected area has enough space
if(xf < xi or yf < yi): 
    sys.exit("ERROR! Selected area isn't big enough for these crops!")

# step for sliding window
stepW = (xf - xi) // 25
stepH = (yf - yi) // 1                     

# sliding window to crop image
h=yi
i=0
while (h < yf):
    w=xi
    while (w < xf):
        # ALWAYS check the place where the crops are going to be stored!!!
        filename = "../crops/%s_%d.bmp" % (fname, i)
        img2 = img.crop((w,h,w+CROP,h+CROP))
        img2.save(filename)
        w = w + stepW
        i = i + 1
    h = h + stepH        