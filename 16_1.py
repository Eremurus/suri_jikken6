from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np
import numpy.linalg as la

m = 3
def phi(x):
    return x

def saisyou_nijou(theta_pre, Phi_pre, x, y):
    phix = np.reshape(phi(x),(m, 1))
    hidari = np.dot(Phi_pre, phix.T)
    theta_pre = np.reshape(theta_pre, (m,1))
    #hidari = np.reshape(hidari, (m, 1))
    #print(hidari)
    migi = np.dot(phix, Phi_pre)
    migi = np.reshape(migi,(m, 1))
    #print(phix.T)
    #print(np.dot(migi, phix.T))
    migi = la.inv(np.eye(m) + np.dot(migi, phix.T))
    #print(migi)
    K = np.dot(hidari, migi)
    #print("Kは",K)
    #print(theta_pre.shape)
    theta = theta_pre + np.dot(K, y - np.dot(phix, theta_pre)) #多分ここが変
    phi_migi = np.dot(K, phi(x))
    phi_migi = np.dot(phi_migi, Phi_pre)
    #print(phi_migi)
    Phi = Phi_pre - phi_migi[0]
    #print(theta)
    #print(theta.T)
    theta_kari = np.reshape(theta.T, (m,))
    #print(theta_kari)
    return theta_kari, Phi

#print(saisyou_nijou(np.array([2.0,2.0,3.0]), 2.0, np.array([1.0,2.0,3.0]),np.array([1.0,2.0,3.0])))

def F(k):
    return 1.0

def y_k_real(y_pre, y_pre_pre, k):
    M, D, K, dt  = 2.0, 1.0, 3.0, 0.01
    w = 1.0
    y_new = (2.0 - (D/M)*dt)*y_pre + (1.0 + D/M*dt - (K/M)*(dt**2)) * y_pre_pre + ((dt**2)/M) * F(k-2) + w
    return y_new

epsilon = 10 **(-6)
y_pre = 0.0
y_pre_pre = 0.0
theta = np.array([0.0,0.0,0.0])
#theta = np.reshape(theta, (m,1))
Phi = 1.0/epsilon

for k in range(10000):
    y = y_k_real(y_pre, y_pre_pre, k)
    x = np.array([y_pre, y_pre_pre, F(k-2)])
    theta, Phi = saisyou_nijou(theta, Phi, x, y)
    y_pre_pre = y_pre
    y_pre = y
    #print(theta)

print(theta)
