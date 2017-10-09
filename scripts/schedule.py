#--------------------------------------------------------------------------
# NOTE: 
# * Make sure that there isn't any other instace of TensorFlow running
#   before calling this script
# 
# * Check if test mode on training.py is not activated
#
# * When ready set the ready flag to True
#--------------------------------------------------------------------------

import os,sys
import numpy as np

# control flag 
try: 
    ready = sys.argv[1].lower() in ['true', 't', 'yes', '1', 'go', 'ready']     
except:
    ready = False

commands = ["python training.py"]

datasets = ["dataset\\kylberg\\training\\"]

testdirs = [""] 

architectures = ["2l_8f_16f_5x5_fc256"]

batches = [512,256,128,64,32]

nruns = 2

for command in commands:
    for data in datasets:
        for arch in architectures:
            for bs in batches:
                for testdir in testdirs:
                    for run in range(0,nruns):
                        # NOTE: adapt runid according with the user's preferences
                        runid = "kylberg_" + arch + "_bs" + str(bs) + "_r" + str(run)
                        new_command = "%s %s %s %d %s %s" % (command,data,arch,bs,runid,testdir)
                        new_command = "%s %s %s %d %s" % (command,data,arch,bs,runid)
                        print(new_command)
                        if(ready): os.system(new_command)

"""
NOTE: Useful methods for generating the run ID

did = data.split("\\") # data ID
did.reverse()
did = did[1]
did = did.split("_")
did.reverse()
did = did[0]
"""