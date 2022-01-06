# create by qilin
# create in 20220105
# PINN model based MPC controller

import numpy as np
from scipy.optimize import minimize
import tensorflow as tf


class Controller:
    # 参数初始化
    def __init__(self, y_set, f, y0, sim_step, contr_step, pred_step):
        self.y_set = y_set  # 控制目标
        self.y_ref = np.ones([contr_step, 2]) * y_set  # 参考轨迹
        self.f = f
        self.y0 = y0
        self.sim_step = sim_step  # 仿真步长
        self.contr_step = contr_step  # 控制步长
        self.pred_step = pred_step  # 预测步长
        self.A = tf.constant([[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0]], dtype=tf.float64)
        self.B = []
        self.C = [0, 0]
        self.u0 = np.ones([self.contr_step, 2])  # 初始化控制量
        self.y_this = np.empty([self.contr_step, 2])  # 初始化每步预测值
        self.loss_recorder = np.empty([0, 1])
        self.k = 5
        self.lambda1 = 0.01

    def optimizer(self):
        fun = lambda x: self.loss(x)
        res = minimize(fun, x0=self.u0, method='SLSQP')
        return res.x

    def optimizer_feedback(self):
        fun = lambda x: self.loss_feedback(x)
        res = minimize(fun, x0=self.u0, method='SLSQP')
        return res.x

    def optimizer_cons(self):
        fun = lambda x: self.loss(x)
        cons = ({'type': 'ineq', 'fun': lambda x: x[0]+1},
                {'type': 'ineq', 'fun': lambda x: x[0]-1},
                {'type': 'ineq', 'fun': lambda x: x[1]+1},
                {'type': 'ineq', 'fun': lambda x: x[1]-1},)
        res = minimize(fun, x0=self.u0, method='SLSQP', constraints=cons)
        # res = minimize(fun, x0=self.u0, method='SLSQP')
        return res.x

    def loss(self, x):
        for i in range(self.contr_step):
            if i == 0:
                self.B = tf.constant(np.hstack([self.C, self.y0, self.sim_step]), dtype=tf.float64)
                self.y_this[i, :] = self.f(x=(tf.matmul(np.mat(x[2*i:2*i+2]), self.A) + self.B))['output_0']  # 第一步输出：x作用在初始状态
            if i != 0:
                self.B = tf.constant(np.hstack([self.C, self.y_this[i-1, :], self.sim_step]), dtype=tf.float64)  # 更新第i步初始状态
                self.y_this[i, :] = self.f(x=(tf.matmul(np.mat(x[2*i:2*i+2]), self.A) + self.B))['output_0']  # 第i步输出：x作用在第i-1作用下的输出

        loss = tf.reduce_mean(tf.square(self.y_this-self.y_ref)) + self.lambda1 * tf.reduce_mean(tf.square(x[0:2]))
        self.loss_recorder = np.vstack([self.loss_recorder, loss])
        return loss

    def loss_feedback(self, x):
        for i in range(self.contr_step):
            if i == 0:
                self.B = tf.constant(np.hstack([self.C, self.y0, self.sim_step]), dtype=tf.float64)
                self.y_this[i, :] = self.f(x=(tf.matmul(np.mat(x[2*i:2*i+2]), self.A) + self.B))['output_0']  # 第一步输出：x作用在初始状态
            if i != 0:
                self.B = tf.constant(np.hstack([self.C, self.y_this[i-1, :], self.sim_step]), dtype=tf.float64)  # 更新第i步初始状态
                self.y_this[i, :] = self.f(x=(tf.matmul(np.mat(x[2*i:2*i+2]), self.A) + self.B))['output_0']  # 第i步输出：x作用在第i-1作用下的输出

        loss = tf.reduce_mean(tf.square(self.y_ref - self.y_this + self.k * (self.y_set*np.ones([self.contr_step, 2]) - self.y0*np.ones([self.contr_step, 2])))) + self.lambda1 * tf.reduce_mean(tf.square(x[0:2]))
        self.loss_recorder = np.vstack([self.loss_recorder, loss])
        return loss