"""
Script to copy/move a directory's deepest files to another directory
""" 

import os
import sys
import shutil
import random

if(len(sys.argv) < 2):
    sys.exit("[ERROR] Must specify the path to root directory!")

try:
    ready = sys.argv[2].lower() in ["true", "t", "yes", "1", "go", "ready"]
except:
    ready = False
    pass

for dirpaths, dirnames, filenames in os.walk(sys.argv[1]):
    # get deepest path which does not contain any sub-directories
    if not dirnames: 
        parts = dirpaths.split(os.sep)
        parts.reverse()
        
        if(parts[0] == ""):
            if(not ready): print(dirpaths, parts[1])
            folder = parts[1]
        else:
            if(not ready): print(dirpaths, parts[0])
            folder = parts[0]
        
        destination = ""
        if(folder.lower() == "empty"):
            destination = "./myPKLot/Empty/"
        elif(folder.lower() == "occupied"):
            destination = "./myPKLot/Occupied/"

        for f in filenames:
            source = os.path.join(dirpaths,f)
            if(not ready): print("\tsource:", source)
            if(not ready): print("\t  dest:", destination)
            
            if(ready): shutil.copy(source,destination)
        
        print(".", end="")