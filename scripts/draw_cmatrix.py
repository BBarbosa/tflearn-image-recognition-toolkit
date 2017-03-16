import matplotlib.pyplot as plt
import sys

#data = sys.stdin.readlines()
#print data

matrix = [
[3025,0,0,0,0,0,0],
[0,2732,0,0,0,0,293],
[0,170,1733,27,0,0,1095],
[0,1,0,2297,0,0,727],
[0,0,0,0,3012,0,13],
[0,0,0,29,0,2996,0],
[0,13,0,0,0,0,3012],
]
           
plt.matshow(matrix)
plt.colorbar()
plt.show()
