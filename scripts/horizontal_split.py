# call example: 
# $ python horizontal_split.py filename.jpg
# ----------------------------- 
from PIL import Image
import sys,os

from colorama import init
from termcolor import colored

# init colored print
init()

if(len(sys.argv) < 2):
    print(colored("Call: $ python scripts\horizontal_split.py {image}","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))
else:
    # example: filename.jpg
    fname        = os.path.splitext(sys.argv[1])[0]    # filename
    extension    = os.path.splitext(sys.argv[1])[1]    # .jpg
    
    img          = Image.open(sys.argv[1]) 
    width,height = img.size

    Hheight = height // 2   # half height for horizontal split

    img2     = img.crop((0,0,width,Hheight))
    filename = "%s_train%s" % (fname,extension)  
    img2.save(filename)

    img3     = img.crop((0,Hheight,width,height))
    filename = "%s_test%s" % (fname,extension)  
    img3.save(filename)
