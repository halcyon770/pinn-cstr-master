# create in 20220104
# CSTR仿真运行

import numpy as np
import math
from scipy.integrate import odeint
from pylab import *
import matplotlib.pyplot as plt

m1 = 0.200
m2 = 0.052
L1 = 0.10
L2 = 0.12
r = 0.20
km = 0.0236
ke = 0.2865
g = 9.8
J1 = 0.004
J2 = 0.001
f1 = 0.01
f2 = 0.001
a = J1 + m2 * r * r
b = m2 * r * L2
c = J2
d = f1 + km * ke
e = (m1 * L1 + m2 * r) * g
f = f2
h = m2 * L2 * g
print(a, b, c, d, e, f, h)


def dflun(y, t, a, b, c, d, e, f, g, h):
    sita1, sita2, w1, w2 = y
    K = np.mat([[0.58498033, -69.40930131, -5.2454833, -8.19545906]])
    xx = np.array([[sita1], [sita2], [w1], [w2]])
    us = np.dot(-K, xx)
    u = us[0, 0]
    dydt = [w1, w2, (
                (-d * c) * w1 + (f * b * math.cos(sita2 - sita1)) * w2 + b * b * math.sin(sita2 - sita1) * math.cos(
            sita2 - sita1) * w1 * w1 - b * c * math.sin(sita1 - sita2) * w2 * w2 + e * c * math.sin(
            sita1) - h * b * math.sin(sita2) * math.cos(sita2 - sita1) + km * c * u) / (
                        a * c - b * b * math.cos(sita1 - sita2) * math.cos(sita2 - sita1)), (
                        (d * b * math.cos(sita1 - sita2)) * w1 - (a * f) * w2 - a * b * math.sin(
                    sita2 - sita1) * w1 * w1 + b * b * math.sin(sita1 - sita2) * math.cos(
                    sita1 - sita2) * w2 * w2 - e * b * math.sin(sita1) * math.cos(sita1 - sita2) + a * h * math.sin(
                    sita2) - b * math.cos(sita1 - sita2) * km * u) / (
                        a * c - b * b * math.cos(sita1 - sita2) * math.cos(sita2 - sita1))]
    return dydt


y0 = [-0.1, 0.05, 0, 0]
t = np.linspace(0, 10)
sol = odeint(dflun, y0, t, args=(a, b, c, d, e, f, g, h))  # 将微分方程解，该函数官方库或者百度有详细的解释
plt.plot(t, sol[:, 0], 'b', label='angle1(t)')
plt.plot(t, sol[:, 1], 'g', label='angle2(t)')
plt.plot(t, sol[:, 2], 'r', label='w1(t)')
plt.plot(t, sol[:, 3], 'y', label='w2(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
