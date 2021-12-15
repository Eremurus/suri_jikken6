import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

n = 2
N = 1000
V_1 = np.array([[100.0, 0.0],[0.0, 1.0]])
V_2 = np.array([[2.0, 0.0],[0.0, 1.0]])

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
        if i < 500:
            phi_v = np.dot(phi(x)[i].T, V_1)
        else:
            phi_v = np.dot(phi(x)[i].T, V_2)
        phi_v_phi = np.dot(phi_v, phi(x)[i])
        phi_v_phi_Phi = np.dot(phi_v_phi, Phi)
        Err_mat += phi_v_phi_Phi
    Err_mat = np.dot(Phi, Err_mat)

    return theta, Err_mat

def calc_theta_6_17(x, y, data_num):
    Phi = np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(data_num):
        if i < 500:
            Q = la.inv(V_1)
        else:
            Q = la.inv(V_2)
        phi_q = np.dot(phi(x)[i].T, Q)
        Phi += np.dot(phi_q, phi(x)[i])
    Phi = la.inv(Phi)

    migi = np.array([0.0,0.0])
    for i in range(data_num):
        if i < 500:
            Q = la.inv(V_1)
        else:
            Q = la.inv(V_2)        
        phi_q = np.dot(phi(x)[i].T, Q)
        migi += np.dot(phi_q, y[i])

    theta = np.dot(Phi, migi)

    Err_mat = np.array([[0.0,0.0],[0.0,0.0]])

    for i in range(data_num):
        if i < 500:
            phi_v = np.dot(phi(x)[i].T, V_1)
        else:
            phi_v = np.dot(phi(x)[i].T, V_2)
        phi_v_phi = np.dot(phi_v, phi(x)[i])
        phi_v_phi_Phi = np.dot(phi_v_phi, Phi)
        Err_mat += phi_v_phi_Phi
    Err_mat = np.dot(Phi, Err_mat)

    return theta, Err_mat

df = pd.read_csv("./suri_jikken6_data/mmse_kadai6.txt",header=None)
data = np.array(df)

x = np.array(data[:,0])
y = np.array(data[:,1:3])

ans_6_5 = calc_theta_6_5(x, y, N)
print(ans_6_5[0])

ans_6_17 = calc_theta_6_17(x, y, N)
print(ans_6_17[0])


theta_0_list = []
theta_1_list = []
theta_2_list = []
theta_3_list = []
N_list = []
real_ans_0_list = []
real_ans_1_list = []
real_ans_2_list = []
real_ans_3_list = []

for k in range(1,10):
    data_num = 2 ** k
    N_list.append(data_num)
    real_ans_0_list.append(3.0)
    real_ans_1_list.append(-2.0)
    real_ans_2_list.append(3.0)
    real_ans_3_list.append(-2.0)

    x = np.array(data[0:data_num,0])
    y = np.array(data[0:data_num,1:3])
    #print(x.shape)
    
    theta_0 = calc_theta_6_5(x, y, data_num)[0][0]
    theta_1 = calc_theta_6_5(x, y, data_num)[0][1]
    theta_2 = calc_theta_6_17(x, y, data_num)[0][0]
    theta_3 = calc_theta_6_17(x, y, data_num)[0][1]

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

'''
[ 3.18835631 -2.09218316]
[ 2.9942022  -2.01469917]
'''