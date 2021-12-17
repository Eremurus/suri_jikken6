import numpy as np
import numpy.linalg as la
import numpy.random as rd

'''
def phi(x):
    return np.array([[x**0, x],[x, x**2],[x**2, x**3]]).T

a = np.array([1,2])
b = np.array([1,2,3])
y = np.array([4,5,6])
a = np.reshape(a, (2,1))

x = np.array([1,2,3])
#print(np.dot(np.dot(a.T, phi(x)), b)-y)
d = np.dot(np.dot(a.T, phi(x)), b)-y
#print(np.sum(d**2))
c = [[5,2,3],[2,3,4],[3,4,5]]
print(c[np.argmin(c)])
'''
#パラメーター
ak = 0.9
ck = 2.0
sigmaV = 1.0
sigmaW = 1.0

v0 = 2.0
mu0 = 3.0

#初期値
theta = rd.normal(mu0, v0)
thetaHat = mu0

y0 = ck*theta + rd.randn()

y = y0
v = v0

#thetaHatなどの準備
THETA = [theta] #theta_0, theta_1,...
THETA_HAT = [thetaHat] #thetaHat_0, thetaHat_1,...

Y = [y0] #y_0, y_1, ...
V = [v0] #v_0, v_1, ...
X = [] #x_1,x_2,...

for i in range(100):
  theta = ak*theta + rd.randn()
  THETA.append(theta)
  y = 2*theta + rd.randn()
  Y.append(y)

  x = ak**2*v + sigmaV**2
  X.append(x)
  v = (sigmaW**2*x)/(ck**2*x + sigmaW**2)
  V.append(v)
  
  F = ck*x/(ck**2*x + sigmaW**2)
  thetaHat = ak*thetaHat + F*(y-ck*ak*thetaHat)
  THETA_HAT.append(thetaHat)

#後退方程式
thetaS = THETA_HAT[100]
vs = V[100]
VS = [vs]
for i in range(100):
  gk = ak*V[99-i]/X[99-i]
  thetaS = THETA_HAT[99-i] + gk*(thetaS - ak*THETA_HAT[99-i])
  vs = V[99-i] + gk**2*(vs-X[99-i])
  VS.append(vs)

#出力
print(THETA[0], thetaS)
print(vs, v0)
print(vs/v0)