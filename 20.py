from numpy.random import random
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def phi(x):
    return np.array([[x**0, x, x**2],[x, x**2, x**3]]).T

def g(alpha, beta, x):
    w = np.random.rand() * 2.0 - 1.0
    ans_tmp = np.dot(alpha, phi(x))
    ans_tmp = np.dot(ans_tmp, beta)
    return ans_tmp + w

epsilon = 10**(-6)
repeat_times = 10
df = pd.read_csv("./suri_jikken6_data/mmse_kadai13.txt",header=None)
data = np.array(df)
x = np.array(data[:,0])
y = np.array(data[:,1])
N = 10000

for time in range(repeat_times):
    alpha = np.array([2.0, -1.5])
    beta = np.array([2.2, 0.0, 3.0])
    alpha_pre = np.array([0.0, 0.0])
    beta_pre = np.array([0.0, 0.0, 0.0])
    j = 0
    while np.sum((alpha_pre - alpha)**2) + np.sum((beta_pre - beta)**2) < epsilon:
        j += 1
        hidari = np.array([[0.0, 0.0],[0.0, 0.0]])
        migi = np.array([0.0, 0.0])
        for i in range(N):
            phi_beta = np.dot(phi(x)[i].T, beta)
            migi += phi_beta * y[i]
            phi_beta = np.reshape(phi_beta, (2,1))
            hidari += np.dot(phi_beta, phi_beta.T)
        hidari = la.inv(hidari)
        alpha = np.dot(hidari, migi)
        
        hidari = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        migi = np.array([0.0, 0.0, 0.0])
        for i in range(N):
            alpha_T_phi = np.dot(alpha, phi(x)[i].T)
            migi += alpha_T_phi.T * y[i]
            alpha_T_phi = np.reshape(alpha_T_phi, (3,1))
            hidari += np.dot(alpha_T_phi.T, alpha_T_phi)
        hidari = la.inv(hidari)
        beta = np.dot(hidari, migi)            

        alpha_pre = alpha
        beta_pre = beta
    print(alpha, beta, j)

'''
(1,-2)
(0.5,-1,2)
'''