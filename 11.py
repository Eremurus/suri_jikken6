import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#基底関数
def phi(x):
    kitei = np.array([x**0, x, x**2, x**3])
    return np.transpose(kitei)

#パラメータを推定する関数
def calc_theta(x, y):
    Phi = la.inv(np.dot(np.transpose(phi(x)),phi(x)))
    ans_kari = np.dot(Phi, np.transpose(phi(x)))
    ans = np.dot(ans_kari, y)
    return ans

#データ読み込み
df = pd.read_csv("./suri_jikken6_data/mmse_kadai4.txt",header=None)
data = np.array(df)
n = 3
N = 1000

x = np.array(data[:,0])
y = np.array(data[:,1])

#パラメータを推定
pred_ans = calc_theta(x, y)
print(pred_ans)