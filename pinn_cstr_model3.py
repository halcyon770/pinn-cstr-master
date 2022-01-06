# create by qilin
# on 2021 12 07

import tensorflow as tf
import datetime, os
import scipy.optimize
import scipy.io
import numpy as np
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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)


# data prep
data = scipy.io.loadmat('cstr_data/sim_luanxu.mat')  	# Load data from file
data = data['sim_luanxu']                                   # 256 points between -1 and 1 [256x1]
data = data[:, [0, 1, 2, 3, 4]]

scaler1 = preprocessing.MinMaxScaler()
data_minMax = scaler1.fit_transform(data)

scaler2 = preprocessing.MinMaxScaler()
x1_minMax = scaler2.fit_transform(data[:, 2:4])

# Domain bounds
lb = 0  # [-1. 0.]
ub = 1  # [1.  0.99]


# Training Data
def trainingdata(N_u, N_f):

    '''Boundary Conditions'''

    # Initial Condition -1 =< x =<1 and t = 0
    all_x0_train = data_minMax[0:12000, :]
    all_x1_train = data_minMax[1:12001, 2:4]

    # choose random N_u points for training
    idx = np.random.choice(all_x0_train.shape[0], N_u, replace=False)

    x0_train = all_x0_train[idx, :]  # choose indices from  set 'idx' (x,t)
    x1_train = all_x1_train[idx, :]


    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    xf_train = lb + (ub-lb)*lhs(5, N_f)
    xf_train = np.vstack((xf_train, x0_train)) # append training points to collocation points

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

    def evaluate(self, x):

        x = (x - lb) / (ub - lb)

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

        nu = 0.01 / np.pi

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
            ca_t = tape.gradient(z[:, 0:1], t)
            T_t = tape.gradient(z[:, 1:2], t)

        del tape

        f_t_inverse = scaler2.inverse_transform(tf.stack([ca_t[:, 0], T_t[:, 0]], axis=1))
        ca_t_inverse = f_t_inverse[:, 0:1]
        T_t_inverse = f_t_inverse[:, 1:2]

        r = 34930800*tf.exp(-11843/(1.985875*T))*ca
        f_ca = (caf - ca) - r - ca_t_inverse
        f_T = (Tf - T) - (-5960/500)*r - (150/500)*(T - Tc) - T_t_inverse

        loss_f = tf.reduce_mean(tf.square(tf.stack([f_ca, f_T], axis=1)))

        return loss_f

    def loss(self, x, y, g):

        loss_u = self.loss_BC(x, y)
        loss_f = self.loss_PDE(g)
        # loss_f = loss_u

        loss = loss_u + lambda1 * loss_f

        return loss

    def optimizer_callback(self, parameters):

        loss_value, loss_u, loss_f = self.loss(x0_train, x1_train, xf_train)

        # u_pred = self.evaluate(X_u_test)
        # error_vec = np.linalg.norm((u - u_pred), 2) / np.linalg.norm(u, 2)
        tf.print(loss_value, loss_u, loss_f)
        # tf.print(loss_value, loss_u, loss_f, error_vec)



N_u = 100  # Total number of data points for 'u'
N_f = 10000  # Total number of collocation points
lambda1 = 1e-4
# Training data
xf_train, x0_train, x1_train = trainingdata(N_u, N_f)

layers = np.array([5, 32, 64, 128, 32, 16, 2])  # 8 hidden layers

PINN = Sequentialmodel(layers)

init_params = PINN.get_weights().numpy()

start_time = time.time()

# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(PINN.loss())

sess = tf.Session()
# train the model with Scipy L-BFGS optimizer
for i in range(1000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step)
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(PINN.loss()))

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))


sim_length = 10000
x = np.empty([sim_length, 2])
u = np.empty([sim_length, 2])
t = np.empty([sim_length, 1])
x_input = np.empty([sim_length, 5])
x[0, :] = data_minMax[0, 2:4]
for i in range(sim_length-1):
    u[i, :] = data_minMax[i, 0:2]
    t[i, :] = data_minMax[i, 4]
    x_input[i, :] = np.hstack([u[i, :], x[i, :], t[i, :]])
    x_pred = PINN.evaluate(x_input[i:i+1, :])
    x[i+1, :] = x_pred

compare = np.empty([sim_length, 2])
compare[:, 0] = x[:, 0]
compare[:, 1] = data_minMax[0:sim_length, 2]
mse = mean_squared_error(compare[:, 0], compare[:, 1])
print(mse)
plt.plot(compare)
plt.title(lambda1)
plt.show()
''' Model Accuracy '''
# u_pred = PINN.evaluate(X_u_test)

# error_vec = np.linalg.norm((u-u_pred),2)/np.linalg.norm(u,2)        # Relative L2 Norm of the error (Vector)
# print('Test Error: %.5f'  % (error_vec))

# u_pred = np.reshape(u_pred, (256, 100), order='F')                        # Fortran Style ,stacked column wise!

''' Solution Plot '''
# solutionplot(u_pred, x0_train, x1_train)
