"""
Script to artificially generate more images through 
geometric transformations like flips, flops, transpose 
and transverse.

Author: bbarbosa
"""

import os
import PIL
import glob
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Script to artificially generate more images through " 
                                             "geometric transformations like flips, flops, transpose "
                                             "and rotates.",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("--folder", required=True, help="<required> images folder")
# optional arguments
parser.add_argument("--flip", required=False, default=False, help="flip up/down (default=False)", type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--flop", required=False, default=False, help="flop left/right (default=False)", type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--rotate", required=False, default=0, help="rotate ROTATE and -ROTATE angle (default=0)", type=int)
parser.add_argument("--rotates", required=False, default=False, help="90, 180 and 270 rotations (default=False)", type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--transp", required=False, default=False, help="transpose (default=False)", type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--all", required=False, default=False, help="make all transformations (default=False)", type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--ext", required=False, default="*", help="images extension (default='*')")

# parse arguments
args = parser.parse_args()
print(args,"\n")

# loads all types of images
images_list = glob.glob(args.folder + "*." + args.ext) 
lil = len(images_list)

print("[INFO] Found %d images" % len(images_list))

for i,image_name in enumerate(images_list):
    try:
        image = Image.open(image_name)
    except:
        continue

    # NOTE: file.txt -> (file , .txt)
    out_name, ext = os.path.splitext(image_name)
    print("Image",i+1,"of",lil,"|",out_name,ext)
    
    if(args.flip or args.all):
        image.transpose(PIL.Image.FLIP_TOP_BOTTOM).save(out_name + "_flip" + ext)
    
    if(args.flop or args.all):
        image.transpose(PIL.Image.FLIP_LEFT_RIGHT).save(out_name + "_flop" + ext)
    
    if(args.rotates or args.all):
        image.transpose(PIL.Image.ROTATE_90).save(out_name + "_r90" + ext)
        image.transpose(PIL.Image.ROTATE_180).save(out_name + "_r180" + ext)
        image.transpose(PIL.Image.ROTATE_270).save(out_name + "_r270" + ext) 

    if(args.rotate or args.all):
        image.rotate(args.rotate).save(out_name + "_r" + str(args.rotate) + ext)
        image.rotate(-args.rotate).save(out_name + "_r-" + str(args.rotate) + ext)

    if(args.transp or args.all):
        image.transpose(PIL.Image.TRANSPOSE).save(out_name + "_transpose" + ext)