from __future__ import division, print_function, absolute_import

import tflearn
import sys,math,time,os,scipy.ndimage,PIL,re,glob,cv2
import numpy as np
from tflearn.data_utils import shuffle,build_image_dataset_from_dir          
from PIL import Image,ImageStat
from colorama import init
from termcolor import colored
from matplotlib import pyplot as plt

# init colored print
init()

numbers = re.compile(r'(\d+)')      # regex for get numbers

def numericalSort(value):
    """
    Splits out any digits in a filename, turns it into an actual 
    number, and returns the result for sorting. Code from
    http://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python

    For directories [1,2,3,10]
    From CMD it gets [1,10,2,3]
    With numerical sort it gets [1,2,3,10]
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# create dataset for HDF5 format
def create_dataset(train_path,height,width,output_path,test_path=None,mode='folder'): 
    train_output_path = test_output_path = None

    print("Creating dataset...")
    if (train_path):
        train_output_path = "%s%s" % (output_path,"_train.h5")
        #build_hdf5_image_dataset(train_path, image_shape=(height,width), mode=mode, output_path=train_output_path, categorical_labels=True, normalize=True, grayscale=False)
        print("\tTrain: ",train_output_path)
    else:
        sys.exit("ERROR: Path to train dataset not set!")
    
    if(test_path):
        test_output_path = "%s%s" % (output_path,"_test.h5")
        #build_hdf5_image_dataset(test_path, image_shape=(height,width), mode=mode, output_path=test_output_path, categorical_labels=True, normalize=True, grayscale=False)
        print("\t Test: ",test_output_path)

    print("Dataset created!\n")
    return train_output_path, test_output_path

# load dataset on format HDF5
def load_dataset(train,height,width,test=None):
    classes = X = Y = Xt = Yt = None

    print("Loading dataset (hdf5)...")
    if(train):
        h5f = h5py.File(train, 'r')
        X = h5f['X'] 
        X = np.array(X)                        # convert to numpy array
        X = np.reshape(X,(-1,height,width,3))  # reshape array to a suitable format
        #X = np.reshape(X,(-1,IMAGE,IMAGE))    # for RNN
        Y = h5f['Y']
        print("\tShape (train): ",X.shape,Y.shape)
        classes = Y.shape[1]
    else:
        sys.exit("ERROR: Path to train dataset not set!")
    
    if(test):
        h5f2 = h5py.File(test, 'r')
        Xt = h5f2['X'] 
        Xt = np.array(Xt)                        # convert to numpy array
        Xt = np.reshape(Xt,(-1,height,width,1))
        #Xt= np.reshape(X,(-1,height,width))     # for RNN  
        Yt = h5f2['Y']
        print("\tShape  (test): ",Xt.shape,Yt.shape)
    
    print("Data loaded!\n")
    return classes,X,Y,Xt,Yt

# image preload (alternative to HDF5)
def load_dataset_ipl(train_path,height,width,test_path=None,mode='folder'):
    classes = X = Y = Xt = Yt = None
    
    print("Loading dataset (image preloader)...")
    if(train_path):
        #X, Y = image_preloader(train_path, image_shape=(width,height), mode=mode, categorical_labels=True, normalize=True)
        classes = Y.shape[1]
    else:
        sys.exit("ERROR: Path to train dataset not set!")

    print("Data loaded!\n")    
    return classes,X,Y,Xt,Yt 

# load images directly from images folder (ex: cropped/5/)
def load_dataset_windows(train_path,height=None,width=None,test=None,shuffled=False,
                         validation=0,mean=False,gray=False,save_dd=False):
    """ 
    Given a folder containing images separated by folders (classes) returns training and testing
    data, if specified.
    """
    classes = X = Y = Xtr = Ytr = Xte = Yte = means_xtr = means_xte = None

    print("Loading dataset (from directory)...")
    if(width and height):
        X,Y = build_image_dataset_from_dir(train_path, resize=(width,height), convert_gray=gray, dataset_file=train_path, 
                                           filetypes=[".bmp",".ppm",".jpg",".png"], shuffle_data=False, categorical_Y=True)
    else:
        X,Y = build_image_dataset_from_dir(train_path, resize=(width,height), convert_gray=gray, dataset_file=train_path, 
                                           filetypes=[".bmp",".ppm",".jpg",".png"], shuffle_data=False, categorical_Y=True)
    try:
        width,height,ch = X[0].shape            # get images dimensions
    except:
        width,height = X[0].shape
        ch = 1
    
    nimages,classes = Y.shape               # get number of images and classes    

    #------------------------------ validation split ------------------------------------------------
    if(validation > 0 and validation <= 1):  # validation = [0,1] float
        counts  = [0] * classes             # create an array to store the number of images per class

        for i in range(0,nimages):
            counts[np.argmax(Y[i])] += 1    # counts the number of images per every class
        
        print("\t       Images: ", nimages)
        print("\t       Counts: ", counts)
        print("\t      Classes: ", classes)

        # show data distribution
        if(save_dd):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            indices = np.arange(len(counts))
            rects = plt.bar(indices,counts)
            plt.xlabel("Class")
            plt.xticks(indices)
            plt.ylabel("Count")
            plt.grid(True)
            image_title = train_path.split("\\")
            image_title.reverse()
            image_title = image_title[1]
            plt.title("Images distribution per class (%s)" % train_path,fontweight='bold')

            for i, v in enumerate(counts):
                ax.text(i - 0.25, v+1, str(v))
    
            plt.savefig("%s.png" % image_title,dpi=300,format='png')

        Xtr = []
        Xte = []
        Ytr = []
        Yte = []

        # split train and test data manually, according to the value of the validation set
        it = 0
        for i in range(0,classes):
            it = 0 
            it += sum(counts[j] for j in range(0,i))
            
            per_class = counts[i]                           # gets the number of images per class
            to_test   = math.ceil(validation * per_class)   # calculates how many images to test per class
            split     = it + per_class - to_test            # calculates the index that splits data in train/test
            
            if False: print("%4d %4d %4d" % (it,split,split+to_test))

            Xtr += X[it:split]
            Ytr = np.concatenate([Ytr, Y[it:split]]) if len(Ytr)>0 else Y[it:split] 
            
            Xte += X[split:split+to_test]
            Yte = np.concatenate([Yte, Y[split:split+to_test]]) if len(Yte)>0 else Y[split:split+to_test]

    #----------------------------------------------------------------------------------------------
    else:
        Xtr = X
        Ytr = Y
    
    del(X)
    del(Y)

    # collect mean pixel value of each image
    if(mean):
        means_xtr = []
        means_xte = []
        
        # training images
        for img in Xtr:
            means_xtr.append(np.mean(img))
        
        means_xtr = np.array(means_xtr)
        means_xtr = np.reshape(means_xtr,(-1,1))

        # validation images
        for img in Xte:
            means_xte.append(np.mean(img))
        
        means_xte = np.array(means_xte)
        means_xte = np.reshape(means_xte,(-1,1))

    Xtr = np.array(Xtr)     # convert train images list to array
    Ytr = np.array(Ytr)     # convert train labels list to array
    Xte = np.array(Xte)     # convert test images list to array
    Yte = np.array(Yte)     # convert test labels list to array

    if(shuffled):
        Xtr,Ytr = shuffle(Xtr,Ytr)      # shuflles training data
        Xte,Yte = shuffle(Xte,Yte)      # shuflles validation data

    Xtr = np.reshape(Xtr,(-1,height,width,ch))      # reshape array to fit on network format
    Xte = np.reshape(Xte,(-1,height,width,ch))      # reshape array to fit on network format

    print("\t         Path: ",train_path)
    print("\tShape (train): ",Xtr.shape,Ytr.shape)
    if(validation > 0): print("\tShape   (val): ",Xte.shape,Yte.shape)
    if(mean):           print("\t        Means: ",means_xtr.shape,means_xte.shape)
    print("Data loaded!\n")

    return classes,Xtr,Ytr,height,width,ch,Xte,Yte,means_xtr,means_xte

# load test images from a directory
def load_test_images(testdir=None,resize=None,mean=False,gray=False,to_array=False):
    """
    Function that loads a set of test images saved by class in distinct folders.
    Returns a list of PIL images an labels.
    """
    image_list = []
    label_list = []
    means_xte = []
    classid = 0

    channels = 3
    if(gray):
        channels = 1
    
    if(testdir is not None):
        print("Loading test images...")
        # get all directories from testdir 
        dirs = sorted(os.walk(testdir).__next__()[1],key=numericalSort)
        
        # for each directory, get all the images inside it (same class)
        for d in dirs:
            #print(colored("\t%s" % d,"yellow"))    # NOTE: just to confirm
            tdir = os.path.join(testdir,d)
            images = os.walk(tdir).__next__()[2]
            
            # for each image, load and append it to the images list 
            for image in images:
                if image.endswith((".bmp",".jpg",".ppm",".png")):
                    image_path = os.path.join(tdir, image)
                    image      = Image.open(image_path)
                    if(gray): image = image.convert('L')
                    if(resize is not None):
                        image = image.resize(resize,Image.ANTIALIAS)
                        #print("\Resized: ",image.size)
                    if(mean is not None):
                        m = np.mean(np.array(image))
                        means_xte.append(m)
                        #print("\t  Mean: ", m)

                    image_list.append(image)
                    label_list.append(classid)
            classid += 1
        
        if(to_array and resize is not None):
            lil = len(image_list)
            image_array  = np.empty((lil,resize[1],resize[0],channels),dtype=np.float32)
            labels_array = np.empty((lil,classid))

            for i in range(lil):
                image_array[i]  = np.array(image_list[i].getdata()).reshape(resize[1],resize[0],channels)
                temp = np.zeros(classid)
                temp[label_list[i]] = 1
                labels_array[i] = temp 
        
        print("\t  Path: ",testdir)
        if(to_array and resize is not None):
            print("\tImages: ",image_array.shape)
            print("\tLabels: ",labels_array.shape)
        else:
            print("\tImages: ",len(image_list))
            print("\tLabels: ",len(label_list))
        
        if(mean):
            means_xte = np.array(means_xte)
            means_xte = np.reshape(means_xte,(-1,1))
            print("\t Mean: ", means_xte.shape)
        
        print("Test images loaded...\n")
    else:
        print(colored("WARNING: Path to test image is not set\n","yellow"))
    
    # NOTE: if needed change to return lists
    if(to_array and resize is not None):
        return image_array,labels_array,means_xte
    else:
        return image_list,label_list,means_xte

# load test images from an index file
def load_test_images_from_index_file(testdir=None,infile=None):
    """
    Function that loads a set of test images according to a indexing file 
    with format: "path_to_image class_id"
    """
    image_list = []
    label_list = []
    index = 0
    
    if(testdir):
        print("Loading test images from index file...")

        try:
            data = np.genfromtxt(infile,delimiter=",",comments='#',names=True, 
                                skip_header=0,autostrip=True)
        except:
            print(colored("WARNING: Index file to test images is not set","yellow"))
            sys.exit(1)
        
        column = data.dtype.names[1] # 'ClassId'

        # picks every sa
        for root, dirs, files in os.walk(testdir):
            for file in files:
                if file.endswith((".bmp",".jpg",".ppm")):
                    image_path = os.path.join(root, file)
                    image      = Image.open(image_path).resize((32,32))

                    #print("Image:", image_path)
                    #print("Label:", int(data[column][index]))
                    #input("")
                    
                    image_list.append(image)
                    label_list.append(int(data[column][index]))
                    index += 1
        
        # NOTE: make it general
        lil = len(image_list) # lenght of image's list
        new_image_list = np.empty((lil,32,32,3),dtype=np.float32)
        for i in range(lil):
            new_image_list[i] = np.array(image_list[i].getdata()).reshape(32,32,3)

        #image_list = np.array(image_list)
        #image_list = np.reshape(image_list,(-1,32,32,3))

        print("\t  Path: ",testdir)
        print("\tImages: ",new_image_list.shape)
        print("\tLabels: ",len(label_list))
        print("Test images loaded...\n")
    else:
        print(colored("WARNING: Path to test images is not set","yellow"))
        
    return new_image_list,label_list

# load an image set from a single folder without subfolders and labels
# TODO: generalize part of reshaping images_list and files extensions
def load_image_set_from_folder(datadir=None,resize=None):
    images_list = []

    print("Loading test images from folder...")
    try:
        filenames = sorted(glob.glob(datadir + "*.jpg"),key=numericalSort)
    except:
        print(colored("WARNING: Couldn't load test images\n","yellow"))
        return None,None

    for infile in filenames:
        img = Image.open(infile)
        if(resize):
            img = img.resize(resize,Image.ANTIALIAS)
        images_list.append(img)
    
    # lenght of image's list
    lil = len(images_list)      
    # NOTE: make it general
    new_images_list = np.empty((lil,64,64,1),dtype=np.float32)
    for i in range(lil):
        # NOTE: make it general
        new_images_list[i] = np.array(images_list[i].getdata()).reshape(64,64,1)
    
    print("\t  Path: ",datadir)
    print("\tImages: ",new_images_list.shape)
    print("Test images loaded...\n")
    
    return new_images_list,filenames

# function to change images colorspace
def convert_images_colorspace(images_array=None,fromc=None,convert_to=None):
    """
    Minimalist function to convert images colorspaces. From RGB 
    to other colorspace (HSV,YCrCb,YUV,...)

    Params:
        `images_array` - images to be converted
        `fromc` - current input images colorspace
        `convert_to` - colorspace that images will be converted
    
    Return: Converted images array if convert_to is a valid
    colorspace.

    TODO: Add option from/to
    """
    new_images_array = images_array

    if(convert_to is not None):
        if(convert_to == 'HSV'):
            ccode = cv2.COLOR_RGB2HSV
        elif(convert_to == 'YCrCb'):
            ccode = cv2.COLOR_RGB2YCrCb
        elif(convert_to == 'YUV'):
            ccode = cv2.COLOR_RGB2YUV
        else:
            print(colored("WARNING: Unknown colorspace! Returned original images."))
            return images_array
        
        lia = len(new_images_array) # length of images array
        for i in range(lia):
            new_images_array[i] = cv2.cvtColor(images_array[i],ccode)
    else:
        print(colored("WARNING: No colorspace selected! Returned original images.","yellow"))

    return new_images_array