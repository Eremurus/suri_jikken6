from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np
import numpy.linalg as la

def phi(x):
    return x

f = open("16_1_theta.txt", "w")

m = 3
def saisyou_nijou(theta, Phi, x, y):
    x_reshape = x.reshape(3,1)
    I = np.eye(3)
    tmp = x_reshape.dot(Phi).dot(x_reshape.T)
    K = Phi * x_reshape.T.dot(la.inv(I + tmp))
    theta = theta + K.dot(y - x.dot(theta))
    Phi = Phi - K.dot(x).dot(Phi)
    return theta, Phi

#print(saisyou_nijou(np.array([2.0,2.0,3.0]), 2.0, np.array([1.0,2.0,3.0]),np.array([1.0,2.0,3.0])))

def F(k):
    return 1.0

def y_k_real(y_pre, y_pre_pre, k):
    M, D, K, dt  = 2.0, 1.0, 3.0, 0.01
    w = np.random.rand()*2.0-1.0
    #w = 0.0
    #print(w)
    y_new = (2.0 - (D/M)*dt)*y_pre + (-1.0 + D/M*dt - (K/M)*(dt**2)) * y_pre_pre + ((dt**2)/M) * F(k-2) + w
    return y_new

epsilon = 10 **(-6)
y_pre = 0.0
y_pre_pre = 0.0
theta = np.array([0.0,0.0,0.0])
#theta = np.reshape(theta, (m,1))
Phi = 1.0/epsilon

for k in range(10000):
    y = y_k_real(y_pre, y_pre_pre, k)
    f.writelines([str(y),"\n"])
    x = np.array([y_pre, y_pre_pre, F(k-2)])
    theta, Phi = saisyou_nijou(theta, Phi, x, y)
    y_pre_pre = y_pre
    y_pre = y
    #f.write(str(theta))
    #print(theta)

f.close()
print(theta)
