"""
Python script for image crop using a square sliding window.

Author: bbarbosa@neadvance.com
Date: 19-03-2018
"""

import os
import sys
import glob
import argparse

from PIL import Image

# argument parser
custom_formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=2000)
parser = argparse.ArgumentParser(description="Python script for cropping images using a square sliding window.", 
                                 prefix_chars='-',
                                 formatter_class=custom_formatter_class)
# required arguments
parser.add_argument("--images_path", required=True, help="<REQUIRED> Path to input images", type=str)
parser.add_argument("--output_path", required=True, help="<REQUIRED> Path to save output images", type=str)
# optional arguments
parser.add_argument("--crop_size", required=False, help="Square crop size (default=64)", type=int, default=64)
parser.add_argument("--crop_step", required=False, help="Crop step based on crop size (deafault=1)", type=int, default=1)
parser.add_argument("--n_crops", required=False, help="Maximum number of crops (deafault=1)", type=int, default=1)
parser.add_argument("--x_initial", required=False, help="Initial X coordinate (deafault=0)", type=int, default=0)
parser.add_argument("--y_initial", required=False, help="Initial Y coordinate (deafault=0)", type=int, default=0)
parser.add_argument("--x_final", required=False, help="Final X coordinate (deafault=0)", type=int, default=0)
parser.add_argument("--y_final", required=False, help="Final Y coordinate (deafault=0)", type=int, default=0)

# parse arguments
args = parser.parse_args()
print(args, "\n")

# accepts regular exceptions as *.png
images_list = glob.glob(args.images_path)
total_images = len(images_list)

image_id = 0

for img_id, image_path in enumerate(images_list):
    print("[INFO] Processing image %d of %d" % (img_id+1, total_images))
    
    crop_id = 0
    parts = image_path.split(os.sep)
    parts.reverse()
    filename, ext = os.path.splitext(parts[0])
    directory = parts[1]

    image = Image.open(image_path)
    
    width, height = image.size

    args.x_final = width
    args.y_final = height

    # check if selected area fits on image
    if(args.x_initial > width  or args.x_final > width  or
       args.x_initial < 0      or args.x_final < 0      or 
       args.y_initial > height or args.y_final > height or
       args.y_initial < 0      or args.y_final < 0):
        print("[ERROR]  Init (%d, %d)" % (args.x_initial, args.y_initial))
        print("[ERROR] Final (%d, %d)" % (args.x_final, args.y_final))
        sys.exit("[ERROR] Selected area gets out of image's dimensions!")

    # to ensure that the crop area doesn't get out of the image
    args.x_final -= args.crop_size     
    args.y_final -= args.crop_size 

    # check that selected area has enough space
    if(args.x_final < args.x_initial or args.y_final < args.y_initial): 
        print("[ERROR]  Init (%d, %d)" % (args.x_initial, args.y_initial))
        print("[ERROR] Final (%d, %d)" % (args.x_final, args.y_final))
        sys.exit("[ERROR] Selected area isn't big enough for these crops!")

    # sliding window step
    stepW = (args.x_final - args.x_initial) // args.n_crops
    stepH = (args.y_final - args.y_initial) // args.n_crops                    

    # sliding window step based one crop_size
    stepW = args.crop_size // args.crop_step
    stepH = args.crop_size // args.crop_step 

    # sliding window to crop image
    h = args.y_initial  # current Height 

    while (h < args.y_final):
        w = args.x_initial # current Width
        while (w < args.x_final):
            # NOTE: ALWAYS check the output directory
            out_path = "%s%s/%s-crop%d.png" % (args.output_path, directory, filename, crop_id)
            print("\t", out_path)
            
            crop = image.crop((w, h, w+args.crop_size, h+args.crop_size))
            crop.save(out_path)
            w = w + stepW
            crop_id += 1
        h = h + stepH     