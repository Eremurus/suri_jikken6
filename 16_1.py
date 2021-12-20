from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np
import numpy.linalg as la

#使ってない
def phi(x):
    return x

f = open("16_1_y.txt", "w")
m = 3

#逐次最小二乗法
def saisyou_nijou(theta, Phi, x, y):
    x = x.reshape(1,3)
    K = Phi.dot(x.T).dot(la.inv(1.0+x.dot(Phi.dot(x.T))))
    #print(K.shape)
    theta = theta + K.dot(y - x.dot(theta))
    Phi = Phi - K.dot(x).dot(Phi)
    return theta, Phi

#外力
def F(k):
    return 1.0

#データ生成
def y_k_real(y_pre, y_pre_pre, k):
    M, D, K, dt  = 2.0, 1.0, 3.0, 0.01
    w = np.random.rand()*2.0-1.0
    #w = 0.0
    #print(w)
    y_new = (2.0 - (D/M)*dt)*y_pre + (-1.0 + D/M*dt - (K/M)*(dt**2)) * y_pre_pre + ((dt**2)/M) * F(k-2) + w
    return y_new

#パラメータ
epsilon = 10 **(-6)
y_pre = 0.0
y_pre_pre = 0.0
theta = np.array([0.0,0.0,0.0])
I = np.eye(m)
Phi = I/epsilon

#実行
for k in range(10001):
    y = y_k_real(y_pre, y_pre_pre, k)
    f.writelines([str(y),"\n"])
    x = np.array([y_pre, y_pre_pre, F(k-2)])
    theta, Phi = saisyou_nijou(theta, Phi, x, y)
    y_pre_pre = y_pre
    y_pre = y

f.close()
print(theta)
