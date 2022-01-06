# create by qilin
# on 2021 12 07
# PINN建模CSTR 用求非线性规划问题的角度训练，可以用所有类型的采样数据

import tensorflow as tf
import datetime, os
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import seaborn as sns
import codecs, json
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import math


matplotlib.rcParams['backend'] = 'SVG'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)


# data prep
data = scipy.io.loadmat('cstr_data/sim_10_u_600000t.mat')  	# Load data from file
data = data['u1u2x1x2tt']
# data_s = data[0:-1, :]
# data_e = data[1:, :]
# data_s[:, [4, 5]] = data_e[:, [4, 5]]
# data = scipy.io.loadmat('cstr_data/sim_luanxu.mat')  	# Load data from file
# data = data['sim_luanxu']
data = data[:, [0, 1, 2, 3, 5, 6, 7]]

# 归一化
data_minMax = np.empty([len(data), 7])
# u1
scaler_caf = preprocessing.MinMaxScaler()
data_minMax[:, 0:1] = scaler_caf.fit_transform(data[:, 0:1])
# u2
scaler_Tc = preprocessing.MinMaxScaler()
data_minMax[:, 1:2] = scaler_Tc.fit_transform(data[:, 1:2])
# x1
scaler_ca = preprocessing.MinMaxScaler()
data_minMax[:, 2:3] = scaler_ca.fit_transform(data[:, 2:3])
K_ca = max(data[:, 2]) - min(data[:, 2])
ca_min = min(data[:, 2])
ca_max = max(data[:, 2])
# x2
scaler_T = preprocessing.MinMaxScaler()
data_minMax[:, 3:4] = scaler_T.fit_transform(data[:, 3:4])
K_T = max(data[:, 3]) - min(data[:, 3])
T_min = min(data[:, 3])
T_max = max(data[:, 3])
# t
scaler_t = preprocessing.MinMaxScaler()
data_minMax[:, 4:5] = scaler_t.fit_transform(data[:, 4:5])
K_t = max(data[:, 4]) - min(data[:, 4])
# y1
scaler_y1 = preprocessing.MinMaxScaler()
data_minMax[:, 5:6] = scaler_y1.fit_transform(data[:, 5:6])
K_y1 = max(data[:, 5]) - min(data[:, 5])
y1_min = min(data[:, 5])
y1_max = max(data[:, 5])
# y2
scaler_y2 = preprocessing.MinMaxScaler()
data_minMax[:, 6:7] = scaler_T.fit_transform(data[:, 6:7])
K_y2 = max(data[:, 6]) - min(data[:, 6])
y2_min = min(data[:, 6])
y2_max = max(data[:, 6])

# Domain bounds
lb = 0  # [-1. 0.]
ub = 1  # [1.  0.99]


# Training Data
def trainingdata(N_u, N_f):

    '''Boundary Conditions'''

    # Initial Condition -1 =< x =<1 and t = 0
    # all_x0_train = data_minMax[0:len(data_minMax), 0:5]
    # all_x1_train = data_minMax[0:len(data_minMax), 5:7]
    all_x0_train = data_minMax[0:30000, 0:5]
    all_x1_train = data_minMax[1:30001, 5:7]

    # choose random N_u points for training
    idx = np.random.choice(all_x0_train.shape[0], N_u, replace=False)
    #
    x0_train = all_x0_train[idx, :]  # choose indices from  set 'idx' (x,t)
    x1_train = all_x1_train[idx, :]

    # x0_train = np.empty([0, 5])
    # x1_train = np.empty([0, 2])
    # for i in range(1):
    #     x0_train = np.vstack([x0_train, all_x0_train[i*6000:i*6000+100, :]])
    #     x0_train = np.vstack([x0_train, all_x0_train[i*6000+7900:(i+1)*8000, :]])
    #     x1_train = np.vstack([x1_train, all_x1_train[i*6000:i*6000+100, :]])
    #     x1_train = np.vstack([x1_train, all_x1_train[i * 6000 + 7900:(i + 1)*8000, :]])

    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    # idx1 = np.random.choice(all_x0_train.shape[0], N_f, replace=False)
    # xf_train = all_x0_train[idx1, 0:5]
    xf_train = all_x0_train[5800:5800+N_f, :]

    # xf_train = lb + (ub-lb)*lhs(5, N_f)
    # xf_train = np.vstack((xf_train, x0_train)) # append training points to collocation points

    return xf_train, x0_train, x1_train


class Sequentialmodel(tf.Module):
    def __init__(self, layers, name=None):

        self.W = []  # Weights and biases
        self.parameters = 0  # total number of parameters

        for i in range(len(layers) - 1):
            input_dim = layers[i]
            output_dim = layers[i + 1]

            # Xavier standard deviation
            std_dv = np.sqrt((2.0 / (input_dim + output_dim)))

            # weights = normal distribution * Xavier standard deviation + 0
            w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv

            w = tf.Variable(w, trainable=True, name='w' + str(i + 1))

            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype='float64'), trainable=True, name='b' + str(i + 1))

            self.W.append(w)
            self.W.append(b)

            self.parameters += input_dim * output_dim + output_dim

    @tf.function
    def __call__(self, x):
        a = x

        for i in range(len(layers) - 2):
            z = tf.add(tf.matmul(a, self.W[2 * i]), self.W[2 * i + 1])
            a = tf.nn.tanh(z)

        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1])  # For regression, no activation to last layer
        return a

    def evaluate(self, x):

        # x = (x - lb) / (ub - lb)

        a = x

        for i in range(len(layers) - 2):
            z = tf.add(tf.matmul(a, self.W[2 * i]), self.W[2 * i + 1])
            a = tf.nn.tanh(z)

        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1])  # For regression, no activation to last layer
        return a

    def get_weights(self):

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array

        for i in range(len(layers) - 1):
            w_1d = tf.reshape(self.W[2 * i], [-1])  # flatten weights
            b_1d = tf.reshape(self.W[2 * i + 1], [-1])  # flatten biases

            parameters_1d = tf.concat([parameters_1d, w_1d], 0)  # concat weights
            parameters_1d = tf.concat([parameters_1d, b_1d], 0)  # concat biases

        return parameters_1d

    def set_weights(self, parameters):

        for i in range(len(layers) - 1):
            shape_w = tf.shape(self.W[2 * i]).numpy()  # shape of the weight tensor
            size_w = tf.size(self.W[2 * i]).numpy()  # size of the weight tensor

            shape_b = tf.shape(self.W[2 * i + 1]).numpy()  # shape of the bias tensor
            size_b = tf.size(self.W[2 * i + 1]).numpy()  # size of the bias tensor

            pick_w = parameters[0:size_w]  # pick the weights
            self.W[2 * i].assign(tf.reshape(pick_w, shape_w))  # assign
            parameters = np.delete(parameters, np.arange(size_w), 0)  # delete

            pick_b = parameters[0:size_b]  # pick the biases
            self.W[2 * i + 1].assign(tf.reshape(pick_b, shape_b))  # assign
            parameters = np.delete(parameters, np.arange(size_b), 0)  # delete

    def loss_BC(self, x, y):

        loss_u = tf.reduce_mean(tf.square(y - self.evaluate(x)))
        return loss_u

    def loss_PDE(self, x_to_train_f):

        g = tf.Variable(x_to_train_f, dtype='float64', trainable=False)
        x0 = g[:, 0:4]
        t = g[:, 4:5]
        Tf = 300
        caf = g[:, 0:1]
        Tc = g[:, 1:2]
        ca = g[:, 2:3]
        T = g[:, 3:4]

        with tf.GradientTape(persistent=True) as tape:
            # tape.watch(x0)
            tape.watch(t)
            g = tf.stack([x0[:, 0], x0[:, 1],x0[:, 2],x0[:, 3], t[:, 0]], axis=1)

            z = self.evaluate(g)
            z_ca_inverse = z[:, 0:1] * K_y1 + y1_min
            z_T_inverse = z[:, 1:2] * K_y2 + y2_min
            ca_t = tape.gradient(z_ca_inverse, t) / K_t
            T_t = tape.gradient(z_T_inverse, t) / K_t

        del tape
        caf_inverse = scaler_caf.inverse_transform(caf)
        Tc_inverse = scaler_Tc.inverse_transform(Tc)
        # ca_inverse = scaler_ca.inverse_transform(ca)
        # T_inverse = scaler_T.inverse_transform(T)

        r = 34930800 * tf.exp(-11843 / (1.985875 * z_T_inverse)) * z_ca_inverse
        f_ca = (caf_inverse - z_ca_inverse) - r - ca_t
        f_T = (Tf - z_T_inverse) - (-5960 / 500) * r - (150 / 500) * (z_T_inverse - Tc_inverse) - T_t
        # f_T = f_T * (K_ca/K_T)

        loss_f = tf.reduce_mean(tf.square(tf.stack([f_ca, f_T], axis=1)))
        # loss_f = tf.bitcast(loss_f, type=tf.float32)

        return loss_f

    def loss(self, x, y, g):

        loss_u = self.loss_BC(x, y)
        loss_f = self.loss_PDE(g)
        # loss_f = loss_u

        loss = loss_u + lambda1 * loss_f

        return loss, loss_u, loss_f

    def optimizerfunc(self, parameters):

        self.set_weights(parameters)

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            loss_val, loss_u, loss_f = self.loss(x0_train, x1_train, xf_train)

        grads = tape.gradient(loss_val, self.trainable_variables)

        del tape

        grads_1d = []  # flatten grads

        for i in range(len(layers) - 1):
            grads_w_1d = tf.reshape(grads[2 * i], [-1])  # flatten weights
            grads_b_1d = tf.reshape(grads[2 * i + 1], [-1])  # flatten biases

            grads_1d = tf.concat([grads_1d, grads_w_1d], 0)  # concat grad_weights
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0)  # concat grad_biases

        return loss_val.numpy(), grads_1d.numpy()

    def optimizer_callback(self, parameters):

        loss_value, loss_u, loss_f = self.loss(x0_train, x1_train, xf_train)

        global loss_record
        loss_record = np.vstack((loss_record, [loss_value, loss_u, loss_f]))
        # u_pred = self.evaluate(X_u_test)
        # error_vec = np.linalg.norm((u - u_pred), 2) / np.linalg.norm(u, 2)
        tf.print(loss_value, loss_u, loss_f)
        # tf.print(loss_value, loss_u, loss_f, error_vec)


N_u = 100  # Total number of data points for 'u'
N_f = 10   # Total number of collocation points
lambda1 = 0
loss_record = np.ones([1, 3])  # 记录loss变化
# Training data
xf_train, x0_train, x1_train = trainingdata(N_u, N_f)

layers = np.array([5, 32, 64, 128, 64, 32, 16, 2])  # N hidden layers

PINN = Sequentialmodel(layers)

init_params = PINN.get_weights().numpy()

start_time = time.time()

# train the model with Scipy L-BFGS optimizer
results = scipy.optimize.minimize(fun=PINN.optimizerfunc,
                                  x0=init_params,
                                  args=(),
                                  method='L-BFGS-B',
                                  jac=True,        # If jac is True, fun is assumed to return the gradient along with the objective function
                                  callback=PINN.optimizer_callback,
                                  options={
                                            'disp': None,
                                            'maxcor': 200,
                                            'ftol': 1 * np.finfo(float).eps,  # The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol
                                            'gtol': 1e-8,
                                            'maxfun':  50000,
                                            'maxiter': 20,
                                            'iprint': -1,   # print update every 50 iterations
                                            'maxls': 50
                                           }
                                  )

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))

print(results)

PINN.set_weights(results.x)
call = PINN.__call__.get_concrete_function(tf.TensorSpec([1, 5], tf.float64))
# tf.saved_model.save(PINN, 'test_save')
tf.saved_model.save(PINN, 'test_save',  signatures=call)

sim_length = len(data)-1000

tau = data_minMax[1, 4] - data_minMax[0, 4]
x0 = np.ones([sim_length, 4]) * data_minMax[0, 0:4]
t = np.empty([sim_length, 1])
for i in range(sim_length):
    t[i, 0:1] = i*tau

x_input = np.hstack([x0, t])

x = PINN.evaluate(x_input)
x1 = PINN.evaluate(data_minMax[0:sim_length, 0:5])
# x = np.empty([sim_length, 2])
# u = np.empty([sim_length, 2])
# t = np.empty([sim_length, 1])
# x_input = np.empty([sim_length, 5])
# x[0, :] = data_minMax[0, 2:4]
# for i in range(sim_length-1):
#     u[i, :] = data_minMax[i, 0:2]
#     t[i, :] = data_minMax[i, 4]
#     x_input[i, :] = np.hstack([u[i, :], x[i, :], t[i, :]])
#     x_pred = PINN.evaluate(x_input[i:i+1, :])
#     x[i+1, :] = x_pred

compare = np.empty([sim_length, 3])
compare[:, 0] = x[0:, 1]
compare[:, 1] = data_minMax[1:sim_length+1, 6]
compare[:, 2] = x1[0:, 1]
mse = mean_squared_error(compare[:, 1], compare[:, 2])
print(mse)
plt.figure(dpi=600)
plt.plot(compare)
plt.title(lambda1)
plt.show()
plt.savefig('plt/test.svg', format='svg')
''' Model Accuracy '''
# u_pred = PINN.evaluate(X_u_test)

# error_vec = np.linalg.norm((u-u_pred),2)/np.linalg.norm(u,2)        # Relative L2 Norm of the error (Vector)
# print('Test Error: %.5f'  % (error_vec))

# u_pred = np.reshape(u_pred, (256, 100), order='F')                        # Fortran Style ,stacked column wise!

''' Solution Plot '''
# solutionplot(u_pred, x0_train, x1_train)
