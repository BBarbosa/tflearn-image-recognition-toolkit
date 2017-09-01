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

# control flag 
try: 
    ready = sys.argv[1].lower() in ['true', 't', 'yes', '1', 'go', 'ready']     
except:
    ready = False

commands = ["python training.py"]

slot_imgs_path = "dataset\\parking\\slots\\"
#datasets = [slot_imgs_path + "slot" + str(s) + "\\" for s in range(13)] + [slot_imgs_path + "all\\"]
datasets = [slot_imgs_path + "all\\"]

testdirs = [""] 

architectures = ["3l_32f_64f_64f_533_fc512"]

batches = [32]

nruns = 3

for command in commands:
    for data in datasets:
        for arch in architectures:
            for bs in batches:
                for testdir in testdirs:
                    for run in range(0,nruns):
                        # NOTE: adapt runid according with the user's preferences
                        slot = data.split("\\")[3]
                        runid = "pklot_" + slot + "_" + arch + "_r" + str(run)
                        new_command = "%s %s %s %d %s %s" % (command,data,arch,bs,runid,testdir)
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