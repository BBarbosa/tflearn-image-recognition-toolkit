"""
NOTE: 
* Make sure that there isn't any other instace of TensorFlow 
  running before calling this script

* Check if test mode on training.py is not activated

* When ready set the ready flag to True
"""

import os
import sys

# ready control flag 
try: 
    ready = sys.argv[1].lower() in ["true", "t", "yes", "1", "go", "ready"]     
except:
    ready = False

operations = ["python training.py"]


train_dirs = ["./dataset/gtsd/train/",
              "./dataset/fabric/training/"]

test_dirs = ["./dataset/gtsd/test/",
             "./dataset/fabric/testing/"]

architectures = ["alexnet", "resnet"]

batches = [256, 128]

params = [""]

cspaces = ["YCrCb", "HSV"]
cspaces = ["YCrCb"]

n_runs = 2

width = height = 32

try:
    # for all operations
    for op in operations:
        # for all train/test pair combination
        for traind,testd in zip(train_dirs, test_dirs):
            # get data ID from path
            data_id = traind.split(os.sep)
            data_id.reverse()
            data_id = data_id[2]
            # for all architectures
            for arch in architectures:
                # for all batch sizes
                for bs in batches:
                    # for all extra parameters
                    for p in params:
                        # for all colorspaces
                        for cs in cspaces:
                            # repeat experience for N times
                            for run in range(0, n_runs):
                                run_id = "%s_%s_bs%d_r%d" % (data_id, arch, bs, run)
                                
                                execute =  "%s --data_dir=%s --arch=%s --run_id=%s " % (op, traind, arch, run_id)
                                execute += "--bsize=%d --width=%d --height=%d "      % (bs, width, height)
                                execute += "--test_dir=%s --param=%s --cspace=%s"    % (testd, p, cs)
                                
                                print(execute)
                                if(ready):
                                    try: 
                                        os.system(execute)
                                    except Exception as e:
                                        print(e)
                                        continue
except Exception as e:
    print(e)