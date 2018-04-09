"""
Python script to extract pre-defined polygon ROIs

Author: Bruno Barbosa
Date: 12-12-2017

TODO: add feature to load from camera
"""

import cv2
import sys
import time
import glob
import yaml
import argparse
import numpy as np

def yaml_parser(filename, skip_header=3, write=None):
    """
    Function to parse YAML formatted files and extract ROIs from it.
    
    Format:
    - { nspot: 1, polyx: [20,121,120,25], polyy: [188,193,222,225] }
    
    Params:
    `filename`    (str) - Path to YAML file.
    `skip_header` (int) - Number of header lines to skip.
    `write`       (str) - Write coords to temporary file ready for copy/paste. 
    
    Returns:
    `points` - List of ROIs object defined by 4 points. Shape (NROIS, 4).
    """

    print("[INFO] Start parsing %s YAML file..." % filename)
    
    with open(filename, 'r') as stream:
        try:
            for _ in range(skip_header):
                _ = stream.readline()
        
            data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print("[ERROR]", exc)
            sys.exit("[ERROR] Error on loading data!")

    # creates temporary file to store coords ready for copy/paste
    if(write is not None):
        ftxt = open(write, "w+")

    # creates data structure to save coords from first coord
    n_rois = len(data)

    if(n_rois > 0):
        n_coords = len(data[0]['polyx'])
        points = np.ndarray(shape=(n_rois, 4), dtype=object)
        if(write is not None):
            ftxt.write("points = np.ndarray(shape=(%d, 4), dtype=object)\n" % n_rois)
    else:
        sys.exit("[ERROR] There is no ROIs to load!")

    # iterate over coords
    for line, elem in enumerate(data):
        polyx = elem['polyx']
        polyy = elem['polyy']

        for column in range(n_coords):
            x = polyx[column]
            y = polyy[column]

            points[line][column] = (x, y)
            
            if(write is not None):    
                ftxt.write("points[%d][%d] = (%d,%d)\n" % (line, column, x, y))

    if(write is not None):
        ftxt.close()
    
    print("[INFO] YAML parsing done!")

    return points

# number of ROIs to exract
NPOLY = 6 * 2

# offset between two parking lines 
offset = 50

# function to create ROIs 
def create_rois(filename=None):
    global NPOLY

    points = yaml_parser(filename, skip_header=3, write=None)

    NPOLY = len(points)

    return points

# function to iterate over all ROIs 
def iterate_over_slots(image, save, imageID, show=False, delay=100, add_border=False):
    # image to draw
    try:
        image_to_draw = image.copy()
    except:
        return None

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
        roi = image[y:y+h, x:x+w].copy()
        
        # add black borders
        if(add_border):
            smask = mask[y:y+h, x:x+w].copy()
            for a in range(h):
                for b in range(w):
                    if(smask[a][b] == 0):
                        roi[a][b] = [0,0,0]
    
        # show ROI to extract
        if(show):
            upsampled_roi = cv2.resize(roi, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('ROI', upsampled_roi)
            box = np.array([points[slot][0], points[slot][1], points[slot][2], points[slot][3]], np.int32)
            box = box.reshape((-1,1,2))
            cv2.polylines(image_to_draw, [box], True, (0,255,0))
            # show image
            cv2.imshow("Image",image_to_draw)

            # check if key was pressed
            key = cv2.waitKey(delay)
            if(key == 27):
                sys.exit("[INFO] Pressed ESC")
            
            if(delay == 0):
                if(key == 98):
                    # b - busy
                    cv2.imwrite('d:/datasets/urban_probe_cameras/agueda_cameras/crops/training/busy/agueda348-img%d-s%d.jpg' % (imageID,slot), roi)
                elif(key == 102):
                    # f - free
                    cv2.imwrite('d:/datasets/urban_probe_cameras/agueda_cameras/crops/training/free/agueda348-img%d-s%d.jpg' % (imageID,slot), roi)

        # save extracted ROI
        if(save):
            # D:\datasets\dst_cameras\crops\camera2
            cv2.imwrite('d:/datasets/urban_probe_cameras/agueda_cameras/crops/agueda348-img%d-s%d.jpg' % (imageID,slot), roi)


# main function
if __name__ == '__main__':
    true_cases = ['true', 't', 'yes', '1']
    parser = argparse.ArgumentParser(description="Automatic pre-defined ROI extractor.", 
                                     prefix_chars='-') 
    
    # optional arguments
    parser.add_argument("--folder", required=False, help="path to images folder (default=None)", type=str)
    parser.add_argument("--camera", required=False, help="use video capture device (default=None)", default=None)
    parser.add_argument("--yaml", required=False, help="Path to .YAML file (default=None)", type=str)
    parser.add_argument("--save", required=False, help="save ROIs (default=False)", default=False, type=lambda s: s.lower() in true_cases)
    parser.add_argument("--show", required=False, help="show ROIS (default=False)", default=False, type=lambda s: s.lower() in true_cases)
    parser.add_argument("--delay", required=False, help="imshow delay (default=100)", default=100, type=int)

    # parse arguments
    args = parser.parse_args()
    print(args, "\n")

    # check either images folder or camera usage
    if(not args.folder and args.camera is None):
        sys.exit("[ERROR] User must specify an image folder or a camera to load images!")

    # load pre-defined ROIs
    if(args.yaml is None):
        sys.exit("[ERROR] No YAML file specified!")
    
    points = create_rois(args.yaml)

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
            iterate_over_slots(image=image, save=args.save, imageID=imageID, show=args.show, delay=args.delay, add_border=False)
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
            iterate_over_slots(image=image, save=args.save, imageID=imageID, show=args.show, delay=args.delay, add_border=False)
            imageID +=1

    print("[INFO] All done!")
            