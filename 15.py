import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def phi(x):
    kitei = np.array([x**0, np.exp(-(x-1.0)**2/2.0), np.exp(-(x+1.0)**2)])
    return np.transpose(kitei)

def calc_theta(x, y):
    Phi = la.inv(np.dot(np.transpose(phi(x)),phi(x)))
    ans_kari = np.dot(Phi, np.transpose(phi(x)))
    ans = np.dot(ans_kari, y)
    return ans

df = pd.read_csv("./suri_jikken6_data/mmse_kadai8.txt",header=None)
data = np.array(df)

N = 6000
M = 4000
n = 3
#始めの6000組
x = np.array(data[0:N,0])
y = np.array(data[0:N,1])
Phi_N = la.inv(np.dot(np.transpose(phi(x)),phi(x)))

N_ans = calc_theta(x, y)
print("前半の推定値:",N_ans)
V_n_hat_kari = y - np.dot(phi(x), N_ans)
V_n_hat_kari = np.dot(V_n_hat_kari, np.transpose(V_n_hat_kari))
V_n_hat_N = V_n_hat_kari / (N - n)
print("前半の推定誤差:",V_n_hat_N)

kari = np.dot(phi(x).T, V_n_hat_N)
Phi_Q_N = la.inv(np.dot(kari, phi(x)))

#後の4000組
x = np.array(data[N:,0])
y = np.array(data[N:,1])
Phi_M = la.inv(np.dot(np.transpose(phi(x)),phi(x)))

M_ans = calc_theta(x, y)
print("後半の推定値:",M_ans)
V_n_hat_kari = y - np.dot(phi(x), M_ans)
V_n_hat_kari = np.dot(V_n_hat_kari, np.transpose(V_n_hat_kari))
V_n_hat_M = V_n_hat_kari / (M - n)
print("後半の推定誤差:",V_n_hat_M)

kari = np.dot(phi(x).T, V_n_hat_M)
Phi_Q_M = la.inv(np.dot(kari, phi(x)))

#合成
Phi1 = Phi_N * V_n_hat_N
Phi2 = Phi_M * V_n_hat_M
gousei_ans = la.inv(la.inv(Phi1)+ la.inv(Phi2)).dot(la.inv(Phi1).dot(N_ans)+la.inv(Phi2).dot(M_ans))
print("合成の推定値:",gousei_ans)
x = np.array(data[:,0])
y = np.array(data[:,1])
ans = calc_theta(x, y)
print("全データの推定値:",ans)

'''
前半の推定値: [ 0.00707684  3.28054335 -2.1908997 ]
前半の推定誤差: 96.86329354733162
後半の推定値: [ 0.09848311  3.10040485 -2.09100212]
後半の推定誤差: 0.010304240740875456
合成の推定値: [ 0.09846859  3.10043371 -2.09101821]
全データの推定値: [ 0.04373209  3.20866562 -2.15117856]
'''
