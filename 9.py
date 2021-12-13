import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def phi(x):
    kitei = np.array([x**0, x, x**2, x**3])
    return np.transpose(kitei)

def calc_theta(x, y):
    Phi = la.inv(np.dot(np.transpose(phi(x)),phi(x)))
    ans_kari = np.dot(Phi, np.transpose(phi(x)))
    ans = np.dot(ans_kari, y)
    return ans

df = pd.read_csv("./suri_jikken6_data/mmse_kadai2.txt",header=None)
data = np.array(df)
n = 4
N = 10000

x = np.array(data[:,0])
y = np.array(data[:,1])

real_ans = calc_theta(x, y)
print(real_ans)

V_n_hat_kari = y - np.dot(phi(x), real_ans)
V_n_hat_kari = np.dot(V_n_hat_kari, np.transpose(V_n_hat_kari))
V_n_hat = V_n_hat_kari / (N - n)
Phi = la.inv(np.dot(np.transpose(phi(x)),phi(x)))
Err_mat = Phi.dot(V_n_hat * x.T.dot(x)).dot(Phi)
print(Err_mat)

theta_0_list = []
theta_1_list = []
theta_2_list = []
theta_3_list = []
N_list = []
real_ans_0_list = []
real_ans_1_list = []
real_ans_2_list = []
real_ans_3_list = []

for i in range(2,14):
    data_num = 2**i
    N_list.append(data_num)
    real_ans_0_list.append(real_ans[0])
    real_ans_1_list.append(real_ans[1])
    real_ans_2_list.append(real_ans[2])
    real_ans_3_list.append(real_ans[3])

    x = np.array(data[0:data_num,0])
    y = np.array(data[0:data_num,1])
    
    theta_0 = calc_theta(x, y)[0]
    theta_1 = calc_theta(x, y)[1]
    theta_2 = calc_theta(x, y)[2]
    theta_3 = calc_theta(x, y)[3]
    theta_0_list.append(theta_0)
    theta_1_list.append(theta_1)
    theta_2_list.append(theta_2)
    theta_3_list.append(theta_3)

fig, ax = plt.subplots(facecolor="w")
ax.plot(N_list, theta_0_list, label="estimate")
ax.plot(N_list, real_ans_0_list, label="true")
ax.legend()

plt.xscale('log')
plt.xlabel('N')
plt.ylabel('theta')
plt.show()

fig, ax = plt.subplots(facecolor="w")
ax.plot(N_list, theta_1_list, label="estimate")
ax.plot(N_list, real_ans_1_list, label="true")
ax.legend()

plt.xscale('log')
plt.xlabel('N')
plt.ylabel('theta')
plt.show()

fig, ax = plt.subplots(facecolor="w")
ax.plot(N_list, theta_2_list, label="estimate")
ax.plot(N_list, real_ans_2_list, label="true")
ax.legend()

plt.xscale('log')
plt.xlabel('N')
plt.ylabel('theta')
plt.show()

fig, ax = plt.subplots(facecolor="w")
ax.plot(N_list, theta_3_list, label="estimate")
ax.plot(N_list, real_ans_3_list, label="true")
ax.legend()

plt.xscale('log')
plt.xlabel('N')
plt.ylabel('theta')
plt.show()

y_bar = np.mean(y)
bunsi = np.sum((np.dot(phi(x),real_ans) - y_bar)**2)
bunbo = np.sum((y - y_bar)**2)
C = bunsi / bunbo
print(C)

#決定変数 0.462430059307111
#[-0.50902942  1.97586067  0.19774405 -0.09866691]
#[[ 3.88685770e-02 -3.02526727e-04 -2.59887092e-03  1.17780810e-05]
#[-3.02526727e-04  4.32490308e-03  1.38561663e-05 -2.43141152e-04]
#[-2.59887092e-03  1.38561663e-05  1.74245886e-04 -4.26806302e-07]
#[ 1.17780810e-05 -2.43141152e-04 -4.26806302e-07  1.36713153e-05]]