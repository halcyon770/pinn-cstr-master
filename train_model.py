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
import pandas as pd
import glob,os
from pyDOE import lhs         #Latin Hypercube Sampling
import seaborn as sns
import codecs, json
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import math
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.models import load_model
import joblib

'''训练集选取'''
class process_data(object):

    def read_train_data(self,root):
        root_path = glob.glob(os.path.join(root,"*"))
        df_u = None
        df_f = None
        for f in root_path[0:num_sample]:
            # print(f)
            cur_df = pd.read_csv(f,header=None)
            new_colums = ["u1","u2","x1","x2","t","y1","y2"]
            cur_df.columns = new_colums

            idx_u = np.linspace(0,10000,N_u)
            idx_u = idx_u.astype(np.int64)  # 将idx_u换成int型
            cur_df_value = cur_df[["u1","u2","x1","x2","t","y1","y2"]].values[:]
            new_u = cur_df_value[idx_u,:]
            cur_u = pd.DataFrame(new_u,columns = new_colums)

            idx_f = []
            for k in range(N_u-1):
                idx_f0 = np.random.randint(idx_u[k],idx_u[k+1],N_f)  # 产生n--m之间的k个整数
                idx_f = np.hstack((idx_f,idx_f0))  # 水平组合
            idx_f = idx_f.astype(np.int64)  # 将idx_f换成int型
            new_f = cur_df_value[idx_f, :]
            cur_f = pd.DataFrame(new_f, columns = new_colums)

            if len(cur_df[["u1","u2","x1","x2","t","y1","y2"]].values) > 0:
                if df_u is None:
                    df_u = cur_u     # 第一组数据；df=[u1,u2,x1,x2,t,y1,y2]
                else:
                    df_u = pd.concat((df_u,cur_u),axis=0)  # 从第二组数据开始纵向拼接；axis=0纵向拼接

            if len(cur_df[["u1","u2","x1","x2","t","y1","y2"]].values) > 0:
                if df_f is None:
                    df_f = cur_f
                else:
                    df_f = pd.concat((df_f,cur_f),axis=0)

        return df_u, df_f

    def get_train_data(self,root):

        df_u,df_f = self.read_train_data(root)
        data_u = df_u.values
        data_f = df_f.values
        return data_u,data_f

'''每次实验需更改的参数'''
train_root = "./data/data_u0"
test_root = 'test_data/test_u1'
model_root = './model/model_test.h5'
xf_mode = 3     # 1——u0不变，x0随机变；
                # 2——x0不变，u0随机变；
                # 3——x0不变，u0随即变，每个u0取同样随机的t
num_sample = 20  # 样本组个数
N_u = 2  # 每个样本组中，网络loss的样本点，至少2个，只能为10000的因数+1
N_f = 1000  # 相邻两个网络loss的样本点间，微分loss的样本点
N_ff = 10000   # Total number of collocation points
lambda1 = 0.001

N_ff_u0 = 40  # xf_mode = 3，u0随机的个数
N_ff_t = 1000  # xf_mode = 3，t随机的个数

p_data = process_data()
train_u_data, train_f_data = p_data.get_train_data(train_root)
# pd.DataFrame(train_data).to_csv('train_data/train0_0')
'''训练集选取结束'''


matplotlib.rcParams['backend'] = 'SVG'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)

# load data
# df = pd.read_csv('cstr_data/all_data2')
# df_np = df.values
# data = df_np[:, 1:]  # 第0列是索引，要去掉

# 归一化
train_data_minMax = np.empty([len(train_u_data), 7])
# u1
caf_max = 15
caf_min = 0
k_caf = caf_max - caf_min
train_data_minMax[:, 0:1] = (train_u_data[:, 0:1] - caf_min)/(caf_max - caf_min)
# scaler_caf = preprocessing.MinMaxScaler()
# train_data_minMax[:, 0:1] = scaler_caf.fit_transform(train_u_data[:, 0:1])
# u2
Tc_max = 350
Tc_min = 250
k_Tc = Tc_max - Tc_min
train_data_minMax[:, 1:2] = (train_u_data[:, 1:2] - Tc_min)/(Tc_max - Tc_min)
# scaler_Tc = preprocessing.MinMaxScaler()
# train_data_minMax[:, 1:2] = scaler_Tc.fit_transform(train_u_data[:, 1:2])
# x1
ca_max = 15
ca_min = 0
k_ca = ca_max - ca_min
train_data_minMax[:, 2:3] = (train_u_data[:, 2:3] - ca_min)/(ca_max - ca_min)
# scaler_ca = preprocessing.MinMaxScaler()
# train_data_minMax[:, 2:3] = scaler_ca.fit_transform(train_u_data[:, 2:3])
# K_ca = max(train_u_data[:, 2]) - min(train_u_data[:, 2])
# ca_min = min(train_u_data[:, 2])
# ca_max = max(train_u_data[:, 2])
# x2
T_max = 350
T_min = 250
k_T = T_max - T_min
train_data_minMax[:, 3:4] = (train_u_data[:, 3:4] - T_min)/(T_max - T_min)
# scaler_T = preprocessing.MinMaxScaler()
# train_data_minMax[:, 3:4] = scaler_T.fit_transform(train_u_data[:, 3:4])
# K_T = max(train_u_data[:, 3]) - min(train_u_data[:, 3])
# T_min = min(train_u_data[:, 3])
# T_max = max(train_u_data[:, 3])
# t
scaler_t = preprocessing.MinMaxScaler()
train_data_minMax[:, 4:5] = scaler_t.fit_transform(train_u_data[:, 4:5])
K_t = max(train_u_data[:, 4]) - min(train_u_data[:, 4])
# y1
y1_max = 15
y1_min = 0
K_y1 = y1_max - y1_min
train_data_minMax[:, 5:6] = (train_u_data[:, 5:6] - y1_min)/(y1_max - y1_min)
# scaler_y1 = preprocessing.MinMaxScaler()
# train_data_minMax[:, 5:6] = scaler_y1.fit_transform(train_u_data[:, 5:6])
# K_y1 = max(train_u_data[:, 5]) - min(train_u_data[:, 5])
# y1_min = min(train_u_data[:, 5])
# y1_max = max(train_u_data[:, 5])
# y2
y2_max = 350
y2_min = 250
K_y2 = y2_max - y2_min
train_data_minMax[:, 6:7] = (train_u_data[:, 6:7] - y2_min)/(y2_max - y2_min)
# scaler_y2 = preprocessing.MinMaxScaler()
# train_data_minMax[:, 6:7] = scaler_y2.fit_transform(train_u_data[:, 6:7])
# K_y2 = max(train_u_data[:, 6]) - min(train_u_data[:, 6])
# y2_min = min(train_u_data[:, 6])
# y2_max = max(train_u_data[:, 6])

# Domain bounds
lb = 0  # [-1. 0.]
ub = 1  # [1.  0.99]


# Training Data
def trainingdata(xf_mode, N_u, N_ff, N_ff_u0, N_ff_t):

    '''Boundary Conditions'''

    # Initial Condition -1 =< x =<1 and t = 0
    # all_x0_train = data_minMax[0:len(data_minMax), 0:5]
    # all_x1_train = data_minMax[0:len(data_minMax), 5:7]
    all_x0_train = train_data_minMax[:, 0:5]
    all_x1_train = train_data_minMax[:, 5:7]

    # choose random N_u points for training
    # idx = np.random.choice(all_x0_train.shape[0], N_u, replace=False)
    #
    x0_train = all_x0_train  # choose indices from  set 'idx' (x,t)
    x1_train = all_x1_train

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
    # xf_train = all_x0_train[5800:5800+N_f, :]

    if xf_mode == 1:
        # constant u0
        u0 = np.ones([N_ff, 2]) * x0_train[0, 0:2]
        xf_train_x0_t = lb + (ub - lb) * lhs(3, N_ff)
        xf_train = np.hstack([u0, xf_train_x0_t])
        xf_train = np.vstack((xf_train, x0_train))

    elif xf_mode == 2:
        # constant x0
        x0 = np.ones([N_ff, 2]) * x0_train[0, 2:4]
        xf_train_u0 = lb + (ub - lb) * lhs(2, N_ff)
        xf_train_t = lb + (ub - lb) * lhs(1, N_ff)
        xf_train = np.hstack([xf_train_u0, x0, xf_train_t])
        xf_train = np.vstack((xf_train, x0_train))

    elif xf_mode == 3:
        # constant x0
        x0 = np.ones([N_ff_u0 * N_ff_t, 2]) * x0_train[0, 2:4]
        xf_train_u0 = lb + (ub - lb) * lhs(2, N_ff_u0)
        xf_train_t = lb + (ub - lb) * lhs(1, N_ff_t)
        xf_train_u0t = np.empty([0, 3])
        for k in range(N_ff_t):  # k从0开始
            xf_train_u0t = np.vstack([xf_train_u0t,
                                     np.hstack([xf_train_u0, np.ones([N_ff_u0, 1]) * xf_train_t[k, :]])])
        xf_train = np.hstack([xf_train_u0t[:, 0:2], x0, xf_train_u0t[:, 2:3]])
        xf_train = np.vstack([xf_train, x0_train])

    # random
    # xf_train = lb + (ub-lb)*lhs(5, N_ff)
    # xf_train = np.vstack((xf_train, x0_train))  # append training points to collocation points

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
        caf_inverse = caf * k_caf + caf_min
        Tc_inverse = Tc * k_Tc + Tc_min
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
        tf.print(len(loss_record), loss_value, loss_u, loss_f)
        # tf.print(loss_value, loss_u, loss_f, error_vec)



# N_u = 100  # Total number of data points for 'u'

loss_record = np.ones([1, 3])  # 记录loss变化
# Training data
xf_train, x0_train, x1_train = trainingdata(xf_mode, N_u, N_ff, N_ff_u0, N_ff_t)

layers = np.array([5, 32, 128, 256, 128, 32, 16, 2])  # N hidden layers

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
                                            'maxiter': 7000,
                                            'iprint': -1,   # print update every 50 iterations
                                            'maxls': 50
                                           }
                                  )

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))

print(results)

PINN.set_weights(results.x)

# tf.saved_model.save(PINN, 'test_save')

'''run model'''
# load test data
df = pd.read_csv(test_root)
df_np = df.values
test_data = df_np[:, 1:]  # 第0列是索引，要去掉
# test data 归一化
test_data_minMax = np.empty([len(test_data), 7])
# u1
test_data_minMax[:, 0:1] = (test_data[:, 0:1] - caf_min)/k_caf
# u2
test_data_minMax[:, 1:2] = (test_data[:, 1:2] - Tc_min)/k_Tc
# x1
test_data_minMax[:, 2:3] = (test_data[:, 2:3] - ca_min)/k_ca
# x2
test_data_minMax[:, 3:4] = (test_data[:, 3:4] - T_min)/k_T
# t
test_data_minMax[:, 4:5] = (test_data[:, 4:5] - scaler_t.data_min_)/scaler_t.data_range_
# y1
test_data_minMax[:, 5:6] = (test_data[:, 5:6] - y1_min)/K_y1
# y2
test_data_minMax[:, 6:7] = (test_data[:, 6:7] - y2_min)/K_y2

sim_length = len(test_data)

tau = test_data_minMax[1, 4] - test_data_minMax[0, 4]  # 归一化后的采样时间
x0 = np.ones([sim_length, 4]) * test_data_minMax[0, 0:4]
t = np.empty([sim_length, 1])
for i in range(sim_length):
    t[i, 0:1] = i*tau

x_input = np.hstack([x0, t])

x = PINN.evaluate(x_input)
x1 = PINN.evaluate(test_data_minMax[0:sim_length, 0:5])
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
compare[:, 1] = test_data_minMax[0:sim_length, 6]
compare[:, 2] = x1[0:, 1]
mse = mean_squared_error(compare[:, 1], compare[:, 2])
print(mse)
plt.figure
plt.plot(compare[:,1:3])
plt.title(lambda1)
plt.show()
# plt.savefig('plt/test.svg', format='svg')