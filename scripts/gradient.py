import numpy as np
import matplotlib.pylab as plt
import cv2
import sys

try:
    img = cv2.imread(sys.argv[1],0)
except:
    exit("ERROR: An image is needed!")

# compute gradient of image
gx, gy = np.gradient(img)
print("gx =", gx)
print("gy =", gy)

# color map
cmap = 'jet'

# plotting
plt.close("all")
plt.figure()
plt.suptitle("Image, and it gradient along each axis")
ax = plt.subplot("221")
ax.axis("off")
ax.imshow(img,cmap=cmap)
ax.set_title("image")

ax = plt.subplot("222")
ax.axis("off")
ax.imshow(gx,cmap=cmap)
ax.set_title("gx")

ax = plt.subplot("223")
ax.axis("off")
ax.imshow(gy,cmap=cmap)
ax.set_title("gy")

ax = plt.subplot("224")
plt.hist(img.ravel(),256,[0,256])
ax.set_title("histogram")
plt.show()