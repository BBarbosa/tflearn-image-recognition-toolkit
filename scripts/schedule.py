# NOTE: Make sure that there isn't any other instace of TensorFlow running

import os

command = ["python training_no_pp.py"]

datasets = ["dataset/ori/side128"]

architectures = ["cifar10"]

for command in commands:
    for arch in architectures:
        for data in datasets:
            for run in range(1):
                # example: data = cropped\genie35\ -> runid = genie35
                runid = data.split("/")[2] + "_" + arch 
                new_command = "%s %s %s %s" % (command,data,arch,runid)
                print("Command: ", new_command)
                os.system(new_command)
