import os,sys,time

command = "python training.py"

#datasets = ["hdf5\genie35.pkl","hdf5\genie.pkl"]

cwd      = os.getcwd()
datasets = ["\\datasets\\ori\\side32","\\datasets\\ori\\side48",
            "\\datasets\\ori\\side64","\\datasets\\ori\\side80"]

num_data = len(datasets)
for i in range(0,num_data):
    datasets[i] = cwd + datasets[i]
    print(datasets[i])

time.sleep(2)

architectures = ["cifar10"]

for arch in architectures:
    for data in datasets:
        #for run in [1,2,3]:
        # example: data = hdf5\genie35.pkl -> runid = genie35
        # runid = data.split("\")[1].split(".")[0] + arch
        runid = data.split("\\")[2] + "_" + arch 
        new_command = "%s %s %s %s" % (command,data,arch,runid)
        print("------------------------------------------------------------")
        print(new_command)
        print("------------------------------------------------------------")
        os.system(new_command)
