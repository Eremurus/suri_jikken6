from os import spawnlp
from numpy.random import random
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def phi(x):
    return np.array([[x**0, x],[x, x**2],[x**2, x**3]]).T

def g(alpha, beta, x):
    w = np.random.rand() * 2.0 - 1.0
    ans_tmp = np.dot(alpha, phi(x))
    ans_tmp = np.dot(ans_tmp, beta)
    return ans_tmp + w

epsilon = 10**(-4)
repeat_times = 10
df = pd.read_csv("./suri_jikken6_data/mmse_kadai13.txt",header=None)
data = np.array(df)
x = np.array(data[:,0])
y = np.array(data[:,1])
N = 10000
gosa_list = []

for time in range(repeat_times):
    a_1 =np.random.rand()*2.0
    a_2 =np.random.rand()*2.0 - 3.0
    b_1 =np.random.rand()*2.0 - 0.5
    b_2 =np.random.rand()*2.0 - 2.0
    b_3 =np.random.rand()*2.0 + 1.0
    alpha = np.array([a_1, a_2])
    beta = np.array([b_1, b_2, b_3])
    alpha_syoki = alpha
    beta_syoki = beta
    alpha_pre = np.array([0.0, 0.0])
    beta_pre = np.array([0.0, 0.0, 0.0])
    j = 0
    
    while np.sum((alpha_pre - alpha)**2) + np.sum((beta_pre - beta)**2) >= epsilon:
        alpha_pre = alpha
        beta_pre = beta
        j += 1
        hidari = np.array([[0.0, 0.0],[0.0, 0.0]])
        migi = np.array([0.0, 0.0])
        for i in range(N):
            phi_beta = np.dot(phi(x)[i], beta)
            migi += phi_beta * y[i]
            phi_beta = np.reshape(phi_beta, (2,1))
            hidari += np.dot(phi_beta, phi_beta.T)
        hidari = la.inv(hidari)
        #print(hidari)
        alpha = np.dot(hidari, migi)
        
        hidari = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        migi = np.array([0.0, 0.0, 0.0])
        #print(migi.shape)

        for i in range(N):
            alpha_reshape = np.reshape(alpha, (2, 1))
            alpha_T_phi = np.dot(alpha_reshape.T, phi(x)[i])
            #print(alpha_T_phi.shape)
            #alpha_T_phi = np.reshape(alpha_T_phi, (3,))
            #print(((alpha_T_phi.T)*y[i]).shape)
            
            #print((alpha_T_phi.T * 3).shape)
            #print(y[i].shape)
            alpha_T_phi_T = np.reshape(alpha_T_phi.T,(3,))
            migi = migi + y[i] * (alpha_T_phi_T)#np.dot((alpha_T_phi.T),y[i])
            #print("migiは",migi)
            
            alpha_T_phi = np.reshape(alpha_T_phi, (1,3))
            #print(alpha_T_phi.shape)
            hidari += np.dot(alpha_T_phi.T, alpha_T_phi)
            #print(hidari)
        #print(hidari)
        hidari = la.inv(hidari)
        beta = np.dot(hidari, migi)
    gosa = np.dot(np.dot(alpha.T, phi(x)), beta) - y
    gosa = np.sum(gosa**2) / N
    print("初期値",alpha_syoki, beta_syoki,"alpha:",alpha,"beta:",beta,"誤差:",gosa)
    gosa_list.append([gosa, alpha, beta, alpha_syoki, beta_syoki])

print(gosa_list[np.argmin(gosa_list)])
'''
(1,-2)
(0.5,-1,2)
初期値 [ 0.88198351 -1.53665416] [ 0.28147475 -1.32918369  2.53393405] alpha: [ 0.87907965 -1.74679254] beta: [ 0.57773367 -1.13074314  2.29569288] 誤差: 0.33289692520721625
初期値 [ 0.7676491  -2.54174922] [ 1.22429426 -0.65419038  2.28962945] alpha: [ 0.78894962 -1.55099889] beta: [ 0.64660782 -1.25410935  2.59422949] 誤差: 0.3329193995801536
初期値 [ 0.62984476 -2.99070649] [ 1.42529549 -0.2928604   2.15821323] alpha: [ 0.81437652 -1.60245924] beta: [ 0.62617943 -1.21545627  2.51018518] 誤差: 0.33291663389390613
初期値 [ 0.13134968 -2.26728033] [-0.37424283 -1.87392872  1.57321468] alpha: [ 1.2131248  -2.37282353] beta: [ 0.42140601 -0.81369574  1.69847641] 誤差: 0.33293760571075365
初期値 [ 0.6294013 -2.312001 ] [ 1.2400193  -1.55985942  1.09836402] alpha: [ 0.79566288 -1.62608227] beta: [ 0.63089796 -1.26200608  2.44527896] 誤差: 0.33292786599258345
初期値 [ 0.29370711 -2.84303114] [ 0.86280628 -1.045895    1.15930158] alpha: [ 1.03220684 -2.11844902] beta: [ 0.48546665 -0.97405249  1.87463964] 誤差: 0.33294237341282557
初期値 [ 1.3178155  -2.58595513] [ 1.21250496 -1.94028981  1.88271072] alpha: [ 0.644844   -1.31297766] beta: [ 0.77965378 -1.55533718  3.03170431] 誤差: 0.33291711140652935
初期値 [ 1.75905833 -1.60943668] [-0.30008309 -1.38187124  2.29619095] alpha: [ 1.10865399 -2.27165929] beta: [ 0.45229525 -0.90644653  1.74902903] 誤差: 0.3329365392120055
初期値 [ 0.49026938 -1.70009918] [ 1.38500097 -0.11870411  1.51993166] alpha: [ 0.9850308  -1.93486129] beta: [ 0.51807312 -1.00408518  2.08010615] 誤差: 0.33292204745785164
初期値 [ 0.99334458 -2.79502105] [ 0.67925781 -0.07801415  1.13280688] alpha: [ 1.69181861 -3.31293935] beta: [ 0.30202616 -0.58377882  1.21604833] 誤差: 0.33293302791851004
'''