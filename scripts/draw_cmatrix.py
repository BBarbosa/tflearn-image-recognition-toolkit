import matplotlib.pyplot as plt
import sys

#data = sys.stdin.readlines()
#print data

matrix = [
[2809,0,0,0,0,0,0],
[0,2809,0,0,0,0,0],
[0,379,2421,0,0,0,9],
[0,325,54,2359,0,0,71],
[0,0,0,0,2809,0,0],
[0,0,1,0,0,2808,0],
[0,0,0,0,0,0,2809],
]
           
plt.matshow(matrix)
plt.colorbar()
plt.show()
