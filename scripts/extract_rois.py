"""
Python script to extract pre-defined polygon ROIs

Author: Bruno Barbosa
Date: 12-12-2017

TODO: add feature to load from camera
"""

import sys
import time
import cv2
import argparse
import glob
import numpy as np

# number of ROIs to exract
NPOLY = 6 * 2

# offset between two parking lines 
offset = 50

# function to create ROIs 
def create_rois(file=None):
    # NOTE: ROIs from the furthest to the closest
    # TODO: add option to load from ROIs from file 

    points = np.ndarray(shape=(NPOLY, 4), dtype=object)
    
    points[0,0] = (3,180) #
    points[0,1] = (103,180)
    points[0,2] = (103,222)
    points[0,3] = (3,222)
    points[1,0] = (103,180) #
    points[1,1] = (203,180)
    points[1,2] = (203,222)
    points[1,3] = (103,222)
    points[2,0] = (203,180) #
    points[2,1] = (303,180)
    points[2,2] = (303,222)
    points[2,3] = (203,222)
    points[3,0] = (303,180) #
    points[3,1] = (403,180)
    points[3,2] = (403,222)
    points[3,3] = (303,222)
    points[4,0] = (403,180) #
    points[4,1] = (503,180)
    points[4,2] = (503,222)
    points[4,3] = (403,222)
    points[5,0] = (503,180) # 
    points[5,1] = (603,180)
    points[5,2] = (603,222)
    points[5,3] = (503,222)

    points[6][0]  = (points[0][0][0],points[0][0][1]+offset)
    points[6][1]  = (points[0][1][0],points[0][1][1]+offset)
    points[6][2]  = (points[0][2][0],points[0][2][1]+offset)
    points[6][3]  = (points[0][3][0],points[0][3][1]+offset)
    points[7][0]  = (points[1][0][0],points[1][0][1]+offset)
    points[7][1]  = (points[1][1][0],points[1][1][1]+offset)
    points[7][2]  = (points[1][2][0],points[1][2][1]+offset)
    points[7][3]  = (points[1][3][0],points[1][3][1]+offset)
    points[8][0]  = (points[2][0][0],points[2][0][1]+offset)
    points[8][1]  = (points[2][1][0],points[2][1][1]+offset)
    points[8][2]  = (points[2][2][0],points[2][2][1]+offset)
    points[8][3]  = (points[2][3][0],points[2][3][1]+offset)
    points[9][0]  = (points[3][0][0],points[3][0][1]+offset)
    points[9][1]  = (points[3][1][0],points[3][1][1]+offset)
    points[9][2]  = (points[3][2][0],points[3][2][1]+offset)
    points[9][3]  = (points[3][3][0],points[3][3][1]+offset)
    points[10][0] = (points[4][0][0],points[4][0][1]+offset)
    points[10][1] = (points[4][1][0],points[4][1][1]+offset)
    points[10][2] = (points[4][2][0],points[4][2][1]+offset)
    points[10][3] = (points[4][3][0],points[4][3][1]+offset)
    points[11][0] = (points[5][0][0],points[5][0][1]+offset)
    points[11][1] = (points[5][1][0],points[5][1][1]+offset)
    points[11][2] = (points[5][2][0],points[5][2][1]+offset)
    points[11][3] = (points[5][3][0],points[5][3][1]+offset)

    return points

# function to iterate over all ROIs 
def iterate_over_slots(image, save, imageID, show=False, delay=100):
    # image to draw
    image_to_draw = image.copy()

    # iterate over all slots
    for slot in np.arange(0, NPOLY):
        # create a copy of the original image to highlight the respective ROI
        copy  = image.copy()
        
        # get the respective ROI coordinates
        polygon = np.array([points[slot][0], points[slot][1], points[slot][2], points[slot][3]])
        
        # draw ROI on image copy with blue color
        copy = cv2.fillConvexPoly(copy, polygon, 255)
        
        # creates mask by filtering ROI polygon
        mask = cv2.inRange(copy, np.array([255, 0, 0]), np.array([255, 0, 0]))
        
        # get bounding box around the ROI polygon
        x, y, w, h = cv2.boundingRect(mask)
        
        # extract desired ROI
        roi = image[y:y+h, x:x+w]
        
        # show ROI to extract
        if(show):
            cv2.imshow('ROI',roi)
            box = np.array([points[slot][0], points[slot][1], points[slot][2], points[slot][3]], np.int32)
            box = box.reshape((-1,1,2))
            cv2.polylines(image_to_draw, [box], True, (0,255,0))
            # show image
            cv2.imshow("Image",image_to_draw)
        
        # save extracted ROI
        if(save):
            cv2.imwrite('./temp/%d.jpg' % imageID, roi)

        # check if key was pressed
        key = cv2.waitKey(delay)
        if(key == 27):
            sys.exit("[INFO] Pressed ESC")

# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automatic pre-defined ROI extractor.", 
                                     prefix_chars='-') 
    
    # optional arguments
    parser.add_argument("--folder", required=False, help="path to images folder", type=str)
    parser.add_argument("--camera", required=False, help="use video capture device (default=None)", default=None)
    parser.add_argument("--save", required=False, help="save ROIs (default=False)", default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument("--show", required=False, help="show ROIS (default=False)", default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])

    # parse arguments
    args = parser.parse_args()
    print(args, "\n")

    # check either images folder or camera usage
    if(not args.folder and args.camera is None):
        sys.exit("[ERROR] User must specify an image folder or a camera to load images!")

    # load pre-defined ROIs
    points = create_rois()

    # counter of current image ID for saving
    imageID = 0

    # /////////////////////// load images from folder ///////////////////////
    if(args.folder):
        # get images paths (add more extensions, if needed)
        images_list  = sorted(glob.glob(args.folder + "*.jpg"))
        images_list += sorted(glob.glob(args.folder + "*.png"))

        # total number of images
        number_of_images = len(images_list)
    
        # iterate over all images paths
        for image_path in images_list:
            # load image from folder
            image = cv2.imread(image_path, 1)
            print("[INFO] Image %d of %d" % (imageID+1, number_of_images))
            iterate_over_slots(image=image, save=args.save, imageID=imageID, show=args.show, delay=100)
            imageID += 1

    # /////////////////////// load images from camera ///////////////////////
    else:
        # initiallize camera 
        try:
            args.camera = int(args.camera)
        except:
            pass

        cam = cv2.VideoCapture(args.camera)

        # get images 
        while(True):
            # load image from camera
            ret_val, image = cam.read()
            print("[INFO] Image %d of -1" % (imageID+1))
            iterate_over_slots(image=image, save=args.save, imageID=imageID, show=args.show, delay=100)
            imageID +=1

    print("[INFO] All done!")
            