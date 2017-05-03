#--------------------------------------------------------------------------
# NOTE: Make sure that there isn't any other instace of TensorFlow running
#       before calling this script
#--------------------------------------------------------------------------

import os

"""
Training schedule
"""

# control flags 
train = True    
test  = True   

commands = ["python training.py"]

datasets = ["dataset/ori/side128"]

architectures = ["cifar10"]

batches = [32]

nruns = 1

for command in commands:
    for data in datasets:
        for arch in architectures:
            for bs in batches:
                for run in range(0,nruns):
                    # NOTE: adapt runid according with the user's preferences
                    runid = data.split("/")[2] + "_" + arch + "_r" + str(run)
                    new_command = "%s %s %s %d %s" % (command,data,arch,bs,runid)
                    print(new_command)
                    if(train): os.system(new_command)

print("")

"""
Testing schedule
NOTE: datasets and architectures must match those used on the training
      NEED to solve test set conflict
"""

commands = ["python autotest.py"]

models = ["models/"]

testdirs = ["%s/../test" % ddir for ddir in datasets]
testdirs = ["dataset/ori/test"]

for command in commands:
    for data,tdir in zip(datasets,testdirs):
        for arch in architectures:
            for model in models:
                # NOTE: adapt runid according with the user's preferences
                runid = "1ke"        
                new_command = "%s %s %s %s %s %s" % (command,data,arch,model,tdir,runid)
                print(new_command)
                if(test): os.system(new_command)