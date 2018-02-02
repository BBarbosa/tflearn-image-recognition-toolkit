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

commands = ["python training.py"]

datasets = ["./dataset/parking/pklot/subset_of_mypklot/training/"]
datasets = ["./dataset/signals/train/"]
datasets = ["./dataset/fabric/side64/training/"]
datasets = ["./dataset/digits/digits_v7/training/"]

testdirs = ["./dataset/parking/pklot/subset_of_mypklot/testing/"] 
testdirs = ["./dataset/signals/test/"]
testdirs = ["./dataset/fabric/side64/testing/"]
testdirs = ["./dataset/digits/digits_v7/testing/"]

architectures = ["myvgg"]
architectures = ["gtsd_1l","gtsd_2l","gtsd_3l","gtsd_4l","gtsd_5l"]

batches = [64]

params = [""]
params = ["64","32","16","08","04","02","01"]
params.reverse()

cspaces = ["YCrCb", "HSV"]
cspaces = [""]

snap = 5

nruns = 1

width = height = 32
width = height = 64

try:
    for command in commands:
        for data in datasets:
            for arch in architectures:
                for bs in batches:
                    for testdir in testdirs:
                        for p in params:
                            for cs in cspaces:
                                for run in range(0,nruns):
                                    runid = "digits_" + arch + "_" + p + "f_fc256_bs" + str(bs) + "_r" + str(run)
                                    execute =  "%s --data_dir=%s --arch=%s --bsize=%d --run_id=%s " % (command, data, arch, bs, runid)
                                    execute += "--width=%d --height=%d --test_dir=%s " % (width, height, testdir)
                                    execute += "--param=%s " % p
                                    print(execute)
                                    if(ready):
                                        try: 
                                            os.system(execute)
                                        except Exception as e:
                                            print(e)
                                            continue
except Exception as e:
    print(e)

"""
NOTE: Useful methods for generating the run ID automatically

did = data.split("\\") # data ID
did.reverse()
did = did[1]
did = did.split("_")
did.reverse()
did = did[0]
"""