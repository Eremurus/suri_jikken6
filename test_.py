import numpy as np
import numpy.linalg as la

a = np.array([1.0, 2.0])
c = np.array([2.0, 3.0])
a = np.reshape(a, (2, 1))
c = np.reshape(c, (2,1))
b = np.dot(a, a.T)
c = np.dot(c, c.T)
#print(b + c)
#print(la.inv(b+c))

def phi(x):
    return np.array([[x**0, x, x**2],[x, x**2, x**3]]).T


a = np.array([1,2,3])
print(phi(a))