from PIL import Image
import sys
import argparse
import os

for root,dirnames,filenames in os.walk(sys.argv[1]):
    if not dirnames:
        print(root)
        for f in filenames:
            if(f.endswith("_color.png") or True):
                print("\t",f)
                full_path = os.path.join(root,f)
                print("\t\t", full_path)
                #img = Image.open(full_path).convert('RGB')
                #img.crop((0,0,320,150)).save(full_path)