# NOTE: Make sure that there isn't any other instace of TensorFlow running

import os

command = "python training.py"

datasets = ["dataset/ori/side8","dataset/ori/side16","dataset/ori/side32",
            "dataset/ori/side48","dataset/ori/side64","dataset/ori/side80",
            "dataset/ori/side128"]

architectures = ["cifar10"]

for arch in architectures:
    for data in datasets:
        #for run in range(1):
        # example: data = cropped\genie35\ -> runid = genie35
        runid = data.split("/")[2] + "_" + arch 
        new_command = "%s %s %s %s" % (command,data,arch,runid)
        print("Command: ", new_command)
        os.system(new_command)
