#--------------------------------------------------------------------------
# NOTE: 
# * Make sure that there isn't any other instace of TensorFlow running
#   before calling this script
# 
# * Check if test mode on training is not activated
#
# * When ready set the ready flag to True
#--------------------------------------------------------------------------

import os,sys

"""
Training schedule
"""

# control flag 
try: 
    ready = sys.argv[1].lower() in ['true', 't', 'yes', '1', 'go', 'ready']     
except:
    ready = False

commands = ["python training.py"]

datasets = ["dataset\\fabric\\side128\\"]
datasets = ["dataset\\signals\\train\\"]

testdirs = [""] 

architectures = ["1l_8f_5x5_fc50"]
architectures = ["cifar10"]

batches = [32]

nruns = 100

for command in commands:
    for data in datasets:
        for arch in architectures:
            for bs in batches:
                for testdir in testdirs:
                    for run in range(0,nruns):
                        # NOTE: adapt runid according with the user's preferences
                        #did = data.split("\\") # data ID
                        #did.reverse()
                        #did = did[1]
                        #did = did.split("_")
                        #did.reverse()
                        #did = did[0]
                        
                        runid = "signals_100runs_" + arch + "_r" + str(run)
                        new_command = "%s %s %s %d %s %s" % (command,data,arch,bs,runid,testdir)
                        print(new_command)
                        if(ready): os.system(new_command)