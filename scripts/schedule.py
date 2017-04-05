# NOTE: Make sure that there isn't any other instace of TensorFlow running

import os

commands = ["python training_nopp.py", "python training_flips.py", "python training_rot.py", 
           "python training_flips_rot.py"]

datasets = ["dataset/ori/side128"]

architectures = ["cifar10"]

for command in commands:
    for arch in architectures:
        for data in datasets:
            for run in range(1):
                # example: data = cropped\genie35\ -> runid = genie35
                runid = command.split(" ")[1].split(".")[0] + "_" + arch 
                new_command = "%s %s %s %s" % (command,data,arch,runid)
                print("Command: ", new_command)
                os.system(new_command)
