from numpy.random import random
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#パラメータ
a_k = 0.9
c_k = 2.0
theta = 3.0
theta_real = np.random.normal(loc=3.0, scale=2.0)
V_k = 2.0
theta_real_list = []
theta_list = []
k_list = []

#Kalman フィルタを繰り返し適用
for k in range(1,101):
    k_list.append(k)
    v_k = np.random.normal()
    theta_real = a_k * theta_real + v_k
    w_k = np.random.normal()
    y_k = c_k * theta_real + w_k
    X_k = (a_k ** 2) * V_k + 1.0
    V_k = 1.0 * X_k / ((c_k**2) * X_k + 1.0)
    F_k = c_k*X_k / ((c_k**2)*X_k + 1.0)
    theta = a_k * theta + F_k * (y_k - c_k * a_k * theta)
    theta_real_list.append(theta_real)
    theta_list.append(theta)

#プロット
fig, ax = plt.subplots(facecolor="w")
ax.plot(k_list, theta_list, label="estimate")
ax.plot(k_list, theta_real_list, label="real theta")
ax.legend()

plt.xlabel("k")
plt.ylabel("theta")
plt.show()
