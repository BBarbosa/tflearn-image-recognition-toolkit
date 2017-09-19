#----------------------------------
# creates N copies of a image
# example: 
# $ python augmentation.py file.jpg
#----------------------------------
from PIL import Image
import sys
from colorama import init
from termcolor import colored

# init colored print
init()

NCOPIES = 300

if(len(sys.argv) < 2):
    print(colored("Call: $ python augmentation.py {image}","yellow"))
    sys.exit(colored("ERROR: Not enough arguments!","yellow"))
else:
    filename = sys.argv[1]
    prefix   = filename.split(".")[0]
    img = Image.open(filename)
    for i in range(0,NCOPIES):
        output = "%s_%d.bmp" % (prefix,i)
        img.save(output,"bmp")