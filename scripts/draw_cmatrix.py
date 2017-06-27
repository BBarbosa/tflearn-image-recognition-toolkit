import matplotlib.pyplot as plt
import sys

#data = sys.stdin.readlines()
#print(data)

matrix = [
[2240,0,0,0,0,0,161],
[0,892,0,0,0,0,1509],
[0,0,0,0,0,1437,964],
[0,0,0,0,0,1178,1223],
[0,0,0,0,155,0,2246],
[0,0,0,0,0,0,2401],
[0,0,0,0,0,0,2401]
]
           
plt.matshow(matrix)
plt.colorbar()
plt.show()
