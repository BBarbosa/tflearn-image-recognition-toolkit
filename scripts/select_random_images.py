"""
Usefull auxiliary script to create a subset of images from a big dataset.

NOTE: Specially adapted to the parking lot example 

Params:
    folder - 
    percentage - 
"""

import os
import sys
import shutil
import random
import argparse


custom_formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=2000)
parser = argparse.ArgumentParser(description="Auxiliary script to create a subset of images from a big dataset.", 
                                 prefix_chars='-',
                                 formatter_class=custom_formatter_class)
# required arguments
parser.add_argument("--folder", required=True, help="<REQUIRED> path to dataset", type=str)
parser.add_argument("--perc", required=True, help="<REQUIRED> percentage for split [0,1]", type=float)

# parse arguments
args = parser.parse_args()
print(args, "\n")

if(args.perc >= 1 or args.perc <= 0):
    sys.exit("[ERROR] Argument 'perc' must be in range ]0,1[!")

TOTAL_BUSY = 337780
TOTAL_FREE = 358071

TOTAL_BUSY = 3375
TOTAL_FREE = 3582

N_BUSY = int(TOTAL_BUSY * args.perc)
N_FREE = int(TOTAL_FREE * args.perc)

for dirpaths, dirnames, filenames in os.walk(args.folder):
    if not dirnames: 
        parts = dirpaths.split(os.sep)
        parts.reverse()
        
        # to deal with "path/to/dataset" and "path/to/dataset/" 
        if(parts[0] == ""):
            print(dirpaths, parts[1])
            folder = parts[1]
        else:
            print(dirpaths, parts[0])
            folder = parts[0]
        
        if(folder.lower() == "free"):
            destination = "./dataset/parking/pklot/subset_of_mypklot/testing/free/"
            # to select N random unique files
            indexes = random.sample(range(0,TOTAL_FREE), N_FREE)
        
        elif(folder.lower() == "busy"):
            destination = "./dataset/parking/pklot/subset_of_mypklot/testing/busy/"
            # to select N random unique files
            indexes = random.sample(range(0,TOTAL_BUSY), N_BUSY)

        for i in indexes:
            source = os.path.join(dirpaths, filenames[i])
            #print("\t|_",source,"->\n",destination)
            shutil.move(source, destination)
        