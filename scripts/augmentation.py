"""
Script to artificially generate more images through 
geometric transformations like flips, flops, transpose 
and transverse.
"""

import argparse,glob,os,PIL
from PIL import Image

parser = argparse.ArgumentParser(description="Script to artificially generate more images through " 
                                             "geometric transformations like flips, flops, transpose "
                                             "and rotates.",
                                 prefix_chars='-') 
# required arguments
parser.add_argument("folder",help="images folder")
# optional arguments
parser.add_argument("--flip",help="flip up/down (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--flop",help="flop left/right (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--rotates",help="rotation (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--transpose",help="transpose (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--all",help="make all transformations (boolean)",type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--extension",help="images extension",default="*")

# parse arguments
args = parser.parse_args()
print(args,"\n")

# loads all types of images
images_list = glob.glob(args.folder + "*." + args.extension) 
lil = len(images_list)

print("[INFO] Found %d images" % len(images_list))

for i,image_name in enumerate(images_list):
    try:
        image = Image.open(image_name)
    except:
        continue

    # NOTE: file.txt -> (file , .txt)
    out_name, ext = os.path.splitext(image_name)
    print("Image",i+1,"of",lil,"|",out_name,ext,)
    
    if(args.flip or args.all):
        image.transpose(PIL.Image.FLIP_TOP_BOTTOM).save(out_name + "_flip" + ext)
    
    if(args.flop or args.all):
        image.transpose(PIL.Image.FLIP_LEFT_RIGHT).save(out_name + "_flop" + ext)
    
    if(args.rotates or args.all):
        image.transpose(PIL.Image.ROTATE_90).save(out_name + "_r90" + ext)
        image.transpose(PIL.Image.ROTATE_180).save(out_name + "_r180" + ext)
        image.transpose(PIL.Image.ROTATE_270).save(out_name + "_r270" + ext) 

    if(args.transpose or args.all):
        image.transpose(PIL.Image.TRANSPOSE).save(out_name + "_transpose" + ext)