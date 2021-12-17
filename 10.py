import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#基底関数
def phi(x):
    return x

#パラメータ推定の関数
def calc_theta(x, y):
    Phi = la.inv(np.dot(np.transpose(phi(x)),phi(x)))
    ans_kari = np.dot(Phi, np.transpose(phi(x)))
    ans = np.dot(ans_kari, y)
    return ans

#データ読み込み
df = pd.read_csv("./suri_jikken6_data/mmse_kadai3.txt",header=None)
data = np.array(df)
n = 2
N = 10000

x = np.array(data[:,0:2])
y = np.array(data[:,2])
y = np.reshape(y, (N,1))

#パラメータを計算
pred_ans = calc_theta(x, y)
print(pred_ans)
theta_0_list = []
theta_1_list = []
N_list = []
real_ans_0_list = []
real_ans_1_list = []

#データ数を変えてパラメータを推定
for i in range(1,14):
    data_num = 2**i
    N_list.append(data_num)
    real_ans_0_list.append(1.5)
    real_ans_1_list.append(2.0)

    x = np.array(data[0:data_num,0:2])
    y = np.array(data[0:data_num,2])

    y = np.reshape(y, (data_num,1))
    theta_0 = calc_theta(x, y)[0][0]
    theta_1 = calc_theta(x, y)[1][0]
    theta_0_list.append(theta_0)
    theta_1_list.append(theta_1)

#プロット
fig, ax = plt.subplots(facecolor="w")
ax.plot(N_list, theta_0_list, label="estimate")
ax.plot(N_list, real_ans_0_list, label="true")
ax.legend()

plt.xscale('log')
plt.xlabel('N')
plt.ylabel('theta')
#グラフの名前
plt.show()

fig, ax = plt.subplots(facecolor="w")
ax.plot(N_list, theta_1_list, label="estimate")
ax.plot(N_list, real_ans_1_list, label="true")
ax.legend()

plt.xscale('log')
plt.xlabel('N')
plt.ylabel('theta')
#グラフの名前
plt.show()

#[[2.3730741 ][1.53731124]]]