"""
NOTE: 
* Make sure that there isn't any other instace of TensorFlow running
  before calling this script

* Check if test mode on training.py is not activated

* When ready set the ready flag to True
"""

import os
import sys

# ready control flag 
try: 
    ready = sys.argv[1].lower() in ['true', 't', 'yes', '1', 'go', 'ready']     
except:
    ready = False

commands = ["python training.py"]

datasets = ["./dataset/signals/train/"]

testdirs = ["./dataset/signals/test/"] 

architectures = ["cifar10"]

batches = [64]

learning_rates = [0.1]

cspaces = ["RGB", "HSV", "YCrCb", "Gray"]

snap = 5

nruns = 1

width = height = 32

try:
    for command in commands:
        for data in datasets:
            for arch in architectures:
                for bs in batches:
                    for testdir in testdirs:
                        for cs in cspaces:
                            for run in range(0,nruns):
                                runid = "signals_" + cs + "_" + arch + "_bs" + str(bs) + "_r" + str(run)
                                execute =  "%s --data_dir=%s --arch=%s --bsize=%d --run_id=%s " % (command, data, arch, bs, runid)
                                execute += "--width=%d --height=%d --test_dir=%s --cspace=%s " % (width, height, testdir, cs)
                                execute += "--snap=%d" % snap
                                print(execute)
                                if(ready):
                                    try: 
                                        os.system(execute)
                                    except:
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