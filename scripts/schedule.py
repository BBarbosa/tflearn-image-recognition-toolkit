#--------------------------------------------------------------------------
# NOTE: Make sure that there isn't any other instace of TensorFlow running
#       before calling this script
#--------------------------------------------------------------------------

import os

commands = ["python training_20e.py", "python training_200e.py", "python training_2000e.py"]

datasets = ["dataset/ori/side128"]

architectures = ["cifar10"]

nruns = 5

for command in commands:
    for arch in architectures:
        for data in datasets:
            for run in range(0,nruns):
                # NOTE: adapt runid according with the user's preferences
                runid = command.split(" ")[1].split(".")[0] + "_" + arch + "_r" + str(run)
                new_command = "%s %s %s %s" % (command,data,arch,runid)
                print("Command: ", new_command)
                #os.system(new_command)
