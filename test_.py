import numpy as np

a = np.array([[3,1],[1,1],[1,1]])
b = np.array([1,2])
print(np.sum(np.sqrt(np.sum((a - b)**2, axis=1))))