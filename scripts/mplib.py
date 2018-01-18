import matplotlib.pyplot as plt
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

import msvcrt

def f():
    global images_list, index, lil
    if(index < 20):
        image = Image.open(images_list[index])
        index += 1
        return image
    else:
        print("[INFO] No more images to show (press any key)")
        ch = msvcrt.getch()
        print("char:", str(ch.decode('ascii')))
        sys.exit(0)

def updatefig(*args):
    im.set_array(f())
    return im,

index = 0
images_list = glob.glob(sys.argv[1])
lil = len(images_list)

fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

fig = plt.gcf()
fig.canvas.set_window_title('My Animation')

im = plt.imshow(f(), animated=True)

try:
    ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
    plt.show()
except Exception as e:
    print(e)