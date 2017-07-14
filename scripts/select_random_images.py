import os,sys,shutil,random
from colorama import init
from termcolor import colored

"""
Usefull auxiliary script to create a subset of images from a big dataset.

NOTE: Specially adapted to the parking lot example 

Params:
    folder - 
    percentage - 
"""

init()
if (len(sys.argv) < 3):
    print(colored("Call: $ python select_random_images.py {folder} {percentage}","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))

folder     = sys.argv[1]            
percentage = float(sys.argv[2])

if(percentage >= 1 or percentage <= 0):
    sys.exit(colored("ERROR: Argument 'percentage' must be in range ]0,1[!","red"))

TOTAL_BUSY = 337780
TOTAL_FREE = 358071

N_BUSY = int(TOTAL_BUSY * percentage)
N_FREE = int(TOTAL_FREE * percentage)

for dirpaths, dirnames, filenames in os.walk(folder):
    if not dirnames: 
        parts = dirpaths.split(os.sep)
        parts.reverse()
        if(parts[0] == ""):
            print(dirpaths,parts[1])
            folder = parts[1]
        else:
            print(dirpaths,parts[0])
            folder = parts[0]
        
        if(folder.lower() == "free"):
            destination = ".\\dataset\\parking\\pklot\\subset_of_mypklot\\free\\"
            # to select N random unique files
            indexes = random.sample(range(0,TOTAL_FREE),N_FREE)
        elif(folder.lower() == "busy"):
            destination = ".\\dataset\\parking\\pklot\\subset_of_mypklot\\busy\\"
            # to select N random unique files
            indexes = random.sample(range(0,TOTAL_BUSY),N_BUSY)

        for i in indexes:
            source = os.path.join(dirpaths,filenames[i])
            #print("\t|_",source,"->\n",destination)
            shutil.copy(source,destination)