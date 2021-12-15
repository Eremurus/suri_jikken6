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

df = pd.read_csv("./suri_jikken6_data/mmse_kadai7.txt",header=None)
data = np.array(df)

N = 6000
M = 4000
#始めの6000組
x = np.array(data[0:N,0])
y = np.array(data[0:N,1])
Phi_N = la.inv(np.dot(np.transpose(phi(x)),phi(x)))
#print(Phi_N.shape)

N_ans = calc_theta(x, y)
print(N_ans)

#後の4000組
x = np.array(data[N:,0])
y = np.array(data[N:,1])
Phi_M = la.inv(np.dot(np.transpose(phi(x)),phi(x)))

M_ans = calc_theta(x, y)
print(M_ans)

#合成
invN = la.inv(Phi_N)
invM = la.inv(Phi_M)
gousei_hidari = la.inv(invM + invN)
gousei_migi = np.dot(invN, N_ans) + np.dot(invM, M_ans)
gousei_ans = np.dot(gousei_migi, gousei_hidari)
print(gousei_ans)

x = np.array(data[:,0])
y = np.array(data[:,1])
ans = calc_theta(x, y)
print(ans)

'''
[-0.00387938  3.0107149  -1.98943435]
[-0.02537589  3.03489937 -1.97772731]
[-0.0124955   3.02042997 -1.98486916]
[-0.0124955   3.02042997 -1.98486916]
'''