from numpy.random import random
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

a_k = 0.9
c_k = 2.0
theta = 3.0
theta_real = np.random.normal(loc=3.0, scale=2.0)
V_k = 2.0
theta_real_list = []
theta_list = []
k_list = []
V_k_list = [V_k]
X_k_list = [0.0]

for k in range(1,101):
    k_list.append(k)
    v_k = np.random.normal()
    theta_real = a_k * theta_real + v_k
    w_k = np.random.normal()
    y_k = c_k * theta_real + w_k
    X_k = (a_k ** 2) * V_k + 1.0
    X_k_list.append(X_k)
    V_k = 1.0 * X_k / ((c_k**2) * X_k + 1.0)
    V_k_list.append(V_k)
    F_k = c_k*X_k / ((c_k**2)*X_k + 1.0)
    theta = a_k * theta + F_k * (y_k - c_k * a_k * theta)
    theta_real_list.append(theta_real)
    theta_list.append(theta)

theta_ks = theta
V_ks = V_k

for i in reversed(range(100)):
    g_k = a_k * V_k_list[i] / X_k_list[i+1]
    theta_ks = theta_list[i] + g_k * (theta_ks - a_k * theta_list[i])
    V_ks = V_k_list[i] + (g_k**2) * (V_ks - X_k_list[i+1])

print(theta_ks)
print(V_ks)
print(V_ks / 2.0)