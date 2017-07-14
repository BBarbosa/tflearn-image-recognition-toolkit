import os,sys,shutil

for dirpaths, dirnames, filenames in os.walk(sys.argv[1]):
    if not dirnames: 
        parts = dirpaths.split(os.sep)
        parts.reverse()
        if(parts[0] == ""):
            print(dirpaths,parts[1])
            folder = parts[1]
        else:
            print(dirpaths,parts[0])
            folder = parts[0]
        
        if(folder.lower() == "empty"):
            destination = ".\\dataset\\parking\\pklot\\mypklot\\free\\"
        elif(folder.lower() == "occupied"):
            destination = ".\\dataset\\parking\\pklot\\mypklot\\busy\\"

        for f in filenames:
            source = os.path.join(dirpaths,f)
            #print("\t|_",source,"->\n",destination)
            shutil.copy(source,destination)