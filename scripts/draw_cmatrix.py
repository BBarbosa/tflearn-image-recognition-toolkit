import matplotlib.pyplot as plt
import sys

#data = sys.stdin.readlines()
#print data

matrix = [
[2401,0,0,0,0,0,0],
[0,2401,0,0,0,0,0],
[0,1081,1320,0,0,0,0],
[0,850,0,1551,0,0,0],
[0,365,0,0,2036,0,0],
[0,10,0,0,0,2391,0],
[0,36,0,0,0,0,2365],
]
           
plt.matshow(matrix)
plt.colorbar()
plt.show()
