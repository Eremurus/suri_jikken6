import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

n = 2
N = 1000
V = np.array([[100.0, 0.0],[0.0, 1.0]])

def phi(x):
    kitei = np.array([[x**0, x**0],[x, x**2]])
    return kitei.T

def calc_theta_6_5(x, y,data_num):
    Phi = np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(data_num):
        Phi += np.dot(phi(x)[i].T, phi(x)[i])
    Phi = la.inv(Phi)
    
    migi = np.array([0.0,0.0])
    for i in range(data_num):
        migi += np.dot(phi(x)[i].T, y[i])
    
    theta = np.dot(Phi, migi)

    Err_mat = np.array([[0.0,0.0],[0.0,0.0]])

    for i in range(data_num):
        phi_v = np.dot(phi(x)[i].T, V)
        phi_v_phi = np.dot(phi_v, phi(x)[i])
        phi_v_phi_Phi = np.dot(phi_v_phi, Phi)
        Err_mat += phi_v_phi_Phi
    Err_mat = np.dot(Phi, Err_mat)

    return theta, Err_mat

def calc_theta_6_17(x, y, data_num):
    Q = la.inv(V)
    Phi = np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(data_num):
        phi_q = np.dot(phi(x)[i].T, Q)
        Phi += np.dot(phi_q, phi(x)[i])
    Phi = la.inv(Phi)

    migi = np.array([0.0,0.0])
    for i in range(data_num):
        phi_q = np.dot(phi(x)[i].T, Q)
        migi += np.dot(phi_q, y[i])

    theta = np.dot(Phi, migi)

    Err_mat = np.array([[0.0,0.0],[0.0,0.0]])

    for i in range(data_num):
        phi_v = np.dot(phi(x)[i].T, V)
        phi_v_phi = np.dot(phi_v, phi(x)[i])
        phi_v_phi_Phi = np.dot(phi_v_phi, Phi)
        Err_mat += phi_v_phi_Phi
    Err_mat = np.dot(Phi, Err_mat)

    return theta, Err_mat

df = pd.read_csv("./suri_jikken6_data/mmse_kadai5.txt",header=None)
data = np.array(df)

x = np.array(data[:,0])
y = np.array(data[:,1:3])

ans_6_5 = calc_theta_6_5(x, y, N)
#print(ans_6_5[0], ans_6_5[1])

ans_6_17 = calc_theta_6_17(x, y, N)
#print(ans_6_17[0], ans_6_17[1])


theta_0_list = []
theta_1_list = []
theta_2_list = []
theta_3_list = []
N_list = []
real_ans_0_list = []
real_ans_1_list = []
real_ans_2_list = []
real_ans_3_list = []

for k in range(1,N):
    N_list.append(k)
    real_ans_0_list.append(3.0)
    real_ans_1_list.append(-2.0)
    real_ans_2_list.append(3.0)
    real_ans_3_list.append(-2.0)

    x = np.array(data[0:k+1,0])
    y = np.array(data[0:k+1,1:3])
    #print(x.shape)
    
    theta_0 = calc_theta_6_5(x, y, k)[0][0]
    theta_1 = calc_theta_6_5(x, y, k)[0][1]
    theta_2 = calc_theta_6_17(x, y, k)[0][0]
    theta_3 = calc_theta_6_17(x, y, k)[0][1]

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
