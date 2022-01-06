# create by qilin
# create in 20220104
# PINN model based MPC sim code

import numpy as np
import math
from scipy.integrate import odeint
from pylab import *
import matplotlib.pyplot as plt
import tensorflow as tf
import nmpc
import pid


# 计算每一个控制量作用下的输出函数
def dflun(y, tau, u, Tf):
    ca, T = y
    r = 34930800 * math.exp(-11843 / (1.985875 * T)) * ca
    dydt = [(u[0] - ca) - r, (Tf - T) - (-5960 / 500) * r - (150 / 500) * (T - u[1])]
    return dydt


# 归一化参数
# u1
caf_max = 15
caf_min = 0
k_caf = caf_max - caf_min
# u2
Tc_max = 350
Tc_min = 250
k_Tc = Tc_max - Tc_min
# x1
ca_max = 15
ca_min = 0
k_ca = ca_max - ca_min
# x2
T_max = 350
T_min = 250
k_T = T_max - T_min
# t
k_t = 10

# 加载模型
PINN = tf.saved_model.load('model_la0.001Nu20Nf401010ma100')
model = PINN.signatures['serving_default']

# 参数初始化
t = 0
sim_step = 0.1
contr_step = 2
pred_step = 1
tau = np.linspace(0, sim_step)
Tf = 300  # 进料温度
y_ref = [4, 310]  # 参考信号
y0 = [2, 300]  # 初始状态
y_sim = np.empty([0, 2])  # 仿真结果
u_sim = np.empty([0, 2])  # 计算控制量

# 归一化
y_ref[0] = (y_ref[0] - ca_min)/k_ca
y_ref[1] = (y_ref[1] - T_min)/k_T
sim_step = sim_step/k_t

# 开始仿真
start_time = time.time()  # 计时
error_recorder = 0

while t < 10:

    # 归一化
    y0[0] = (y0[0] - ca_min) / k_ca
    y0[1] = (y0[1] - T_min) / k_T
    # controller = nmpc.Controller(y_ref, model, y0, sim_step, contr_step, pred_step)
    # u = controller.optimizer_feedback()  # MPC计算控制量
    # u = u[0:2]  # 取第一个作用到系统

    error_recorder += np.array(y_ref) - np.array(y0)
    pid_controller = pid.PID(y_ref, y0, sim_step, error_recorder)
    u = pid_controller.controller()

    # 反归一化
    u[0] = u[0] * k_caf + caf_min
    u[1] = u[1] * k_Tc + Tc_min
    y0[0] = y0[0] * k_ca + ca_min
    y0[1] = y0[1] * k_T + T_min

    y_sim_this = odeint(dflun, y0, tau, args=(u, Tf))  # 仿真
    y0 = y_sim_this[-1]  # 初始状态更新
    y_sim_this = y_sim_this[0:-1]  # 舍去最后一个数据
    u_this = u * np.ones([len(y_sim_this), 2])  # 数据保存
    y_sim = np.vstack([y_sim, y_sim_this])
    u_sim = np.vstack([u_sim, u_this])
    t = t + sim_step * k_t  # 时间前进

elapsed = time.time() - start_time  # 计算结束
print('Calculation time: %.2f' % elapsed)

# 画图
plt.figure(dpi=600)
plt.plot(y_sim[:, 0], 'r', label='ca(t)')
plt.plot(u_sim[:, 0], 'y', label='caf(t)')
# plt.plot(y_sim[:, 1], 'g', label='T(t)')
# plt.plot(u_sim[:, 1], 'b', label='Tc(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
