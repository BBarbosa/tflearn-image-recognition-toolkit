# call example: 
# $ python vertical_split.py filename.jpg
# ----------------------------- 
from PIL import Image
import sys,os

from colorama import init
from termcolor import colored

# init colored print
init()

if(len(sys.argv) < 2):
    print(colored("Call: $ python scripts\\vertical_split.py {image}","yellow"))
    sys.exit(colored("ERROR: Not enough arguments!","yellow"))
else:
    # example: filename.jpg
    fname        = os.path.splitext(sys.argv[1])[0]    # filename
    extension    = os.path.splitext(sys.argv[1])[1]    # .jpg

    img          = Image.open(sys.argv[1])               
    width,height = img.size

    Wwidth = width // 2   # half width for vertical split

    img2     = img.crop((0,0,Wwidth,height))
    filename = "%s_train%s" % (fname,extension)  
    img2.save(filename)

    img3     = img.crop((Wwidth,0,width,height))
    filename = "%s_test%s" % (fname,extension)  
    img3.save(filename)
