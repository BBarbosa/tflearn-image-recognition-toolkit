"""
Python module to load coordinates from YAML file to an object.

- { nspot: 1, polyx: [20,121,120,25], polyy: [188,193,222,225] }
   
to 

points[0][1] = (20, 188)
points[0][2] = (121, 193)
points[0][3] = (120, 222)
points[0][4] = (25, 225)
"""

import sys
import yaml
import argparse   
import numpy as np


# /////////////////////////////////////////////////////////
#                     Global variables
# /////////////////////////////////////////////////////////
DEBUG = False   # Debug flag
# /////////////////////////////////////////////////////////


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

# /////////////////////////////////////////////////////
#                    Main method
# /////////////////////////////////////////////////////
if __name__ == "__main__":
    # argument parser
    custom_formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=2000)
    parser = argparse.ArgumentParser(description="Python YAML parser.", 
                                     prefix_chars='-',
                                     formatter_class=custom_formatter_class)
    # required arguments
    parser.add_argument("--yaml_path", required=True, help="<REQUIRED> Path to YAML file to parse", type=str)
    # optional arguments
    parser.add_argument("--skip_header", required=False, help="Path to .YAML file that stores annotations (default=3)", default=3, type=int)
    parser.add_argument("--txt_file", required=False, help="Path to .TXT file that stores text ready for copy/paste", type=str)
    parser.add_argument("--debug", required=False, help="Enable debug mode", nargs="?", default=argparse.SUPPRESS)
    
    # parse arguments
    args = parser.parse_args()
    print(args, "\n")

    try:
        DEBUG = args.debug
        DEBUG = True
    except:
        DEBUG = False 

    points = yaml_parser(args.yaml_path, skip_header=args.skip_header, write=args.txt_file)
    
    if(DEBUG):
        print("--------------")
        print("[DEBUG] Points")
        print(points)