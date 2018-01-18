"""
Script to copy/move a directory's deepest files to another directory
""" 

import os,sys,shutil,random

for dirpaths, dirnames, filenames in os.walk(sys.argv[1]):
    if not dirnames: 
        parts = dirpaths.split(os.sep)
        parts.reverse()
        if(parts[0] == ""):
            print(dirpaths,parts[2])
            folder = parts[2]
        else:
            print(dirpaths,parts[1])
            folder = parts[1]
        
        destination = ""
        if(folder.lower() == "female"):
            destination = "./dataset/faces/gender/female/"
        elif(folder.lower() == "male"):
            destination = "./dataset/faces/gender/male/"

        for f in random.sample(filenames,3):
            source = os.path.join(dirpaths,f)
            print("\t|_",source,"->",destination)
            shutil.copy(source,destination)