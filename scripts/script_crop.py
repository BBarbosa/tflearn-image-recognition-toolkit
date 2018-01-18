from __future__ import division, print_function, absolute_import
import os
import sys
import platform

from colorama import init
from termcolor import colored

# init colored print
init()

# check number of arguments
if (len(sys.argv) < 2):
    print(colored("Call: $ python scripts\script_crop.py {data_path}","yellow"))
    sys.exit(colored("ERROR: Not enough arguments!","yellow"))

# specify OS
OS = platform.system()

# clear screen and show OS
if(OS == 'Windows'):
    os.system('cls')
    command = "python ..\..\..\scripts\crop.py"
    joint   = "\\" 
else:
    os.system('clear')
    command = "python ../../../scripts/crop.py"
    joint   = "/"
print("Operating System --> %s\n" % OS)

# $1 = dataset/5/train/ OR dataset\5\train\ 
mydir  = sys.argv[1]
cwd    = os.getcwd()
newdir = os.path.join(cwd,mydir)
os.chdir(newdir)  # change to train data directorys
print('CWD: ',os.getcwd(),'\n')

# Not working !!! Fix 
for root, dirs, files in os.walk(newdir):
    for f in files:
        if(True or f.endswith(".png")):
            image = os.path.join(root, f)
            print("Cropping file:",image)
            new_command = "%s %s" % (command,image)
            print("\tCommand:",new_command)
            os.system(new_command)
