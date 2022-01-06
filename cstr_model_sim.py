# create in 20220104
# CSTR仿真运行

import numpy as np
import math
from scipy.integrate import odeint
from pylab import *
import matplotlib.pyplot as plt

caf = 5
Tc = 310
Tf = 300
k = 0

def dflun(y, t, caf, Tc, Tf):
    ca, T = y
    r = 34930800 * math.exp(-11843 / (1.985875 * T)) * ca
    dydt = [(caf - ca) - r, (Tf - T) - (-5960 / 500) * r - (150 / 500) * (T - Tc)]
    global k
    k = k + 1
    return dydt


y0 = [2, 280]
t = np.linspace(0, 0.01)
sol = odeint(dflun, y0, t, args=(caf, Tc, Tf))  # 将微分方程解，该函数官方库或者百度有详细的解释
plt.plot(t, sol[:, 0], 'b', label='ca(t)')
plt.plot(t, sol[:, 1], 'g', label='T(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
