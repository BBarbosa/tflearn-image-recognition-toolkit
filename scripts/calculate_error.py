from __future__ import division, print_function, absolute_import
import os,sys,platform,subprocess,time

from colorama import init
from termcolor import colored

# init colored print
init()

# check number of arguments
if (len(sys.argv) < 4):
    print(colored("Call: $ python scripts\calculate_error.py {test_path} {architecture} {model}","red"))
    sys.exit(colored("ERROR: Not enough arguments!","red"))
else:
    # specify OS
    OS = platform.system() 
    
    # clear screen and show OS
    if(OS == 'Windows'):
        import winsound as ws
        os.system('cls')
    else:
        os.system('clear')
    print("Operating System: %s\n" % OS)

    # variable to ID each class
    classid   = -1
    
    mydir = sys.argv[1]     # $1 = dataset/fabric/test/
    arch  = sys.argv[2]     # $2 = architecture name
    model = sys.argv[3]     # $3 = path to trained model
    
    #command = "python classify.py"
    #command = "python classify_sw.py"
    command = "python utils\communicator.py"

    comm = False
    if(command == "python utils\communicator.py"):
        comm = True
        # initialize subprocess by calling the classification script
        subprocess.Popen(['python','classify_sv.py',arch,model])

    time.sleep(8) # to ensure that it loads the trained network (is it enough?)

    for root, dirs, files in os.walk(mydir):
        for file in files:
            # check image format!!!
            if file.endswith((".jpg",".bmp")):
                test_image = os.path.join(root, file)
                print("File:",test_image)

                if(comm):
                    new_command = "%s %s %d" % (command,test_image,classid)
                else:
                    new_command = "%s %s %s %s %d" % (command,arch,model,test_image,classid)

                print("\tCommand:",new_command)
                os.system(new_command)
        classid += 1

    if(OS == 'Windows'):
        freq = 1500
        dur  = 1500 
        ws.Beep(freq,dur)