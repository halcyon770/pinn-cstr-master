# create by qilin
# pinn for cstr, data with 1 single input (control)

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from sklearn import preprocessing
import scipy.io
from pyDOE import lhs
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
print(tf.__version__)
print(tf.test.is_gpu_available())
np.seterr(divide='ignore', invalid='ignore')

# data prep
# data = scipy.io.loadmat('cstr_data/u1u2x1x2tt(2).mat')  	# Load data from file
# data = data['u1u2x1x2tt']
# data = scipy.io.loadmat('cstr_data/sim_luanxu.mat')  	# Load data from file
# data = data['sim_luanxu']
data = scipy.io.loadmat('cstr_data/sim_only10000_1.mat')  	# Load data from file
data = data['u1u2x1x2tt']
data = data[:, [0, 1, 2, 3, 4]]


# 归一化
data_minMax = np.empty([len(data), 5])
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
T_max = max(data[:, 4])
# t
scaler_t = preprocessing.MinMaxScaler()
data_minMax[:, 4:5] = scaler_t.fit_transform(data[:, 4:5])
K_t = max(data[:, 4]) - min(data[:, 4])

# Domain bounds
lb = 0  # [-1. 0.]
ub = 1  # [1.  0.99]

# active function
activation = 'tanh'


# Training Data
def trainingdata(N_u, N_f):

    '''Boundary Conditions'''

    # Initial Condition -1 =< x =<1 and t = 0
    all_x0_train = data_minMax[0:len(data_minMax)-1, :]
    all_x1_train = data_minMax[1:len(data_minMax), 2:4]

    # choose random N_u points for training
    idx = np.random.choice(all_x0_train.shape[0], N_u, replace=False)

    x0_train = all_x0_train[idx, 2:5]  # choose indices from  set 'idx' (x,t)
    x1_train = all_x1_train[idx, :]


    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    idx1 = np.random.choice(all_x0_train.shape[0], N_f, replace=False)
    xf_train = all_x0_train[idx1, :]
    # xf_train = all_x0_train[0:N_f]

    # xf_train = lb + (ub-lb)*lhs(5, N_f)
    # xf_train = np.vstack((xf_train, x0_train)) # append training points to collocation points

    return xf_train, x0_train, x1_train


# 构建神经网络，也可以用上面说的Sequential容器构建，不过用自定义方式，更灵活
class Network(keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        # 创建 N 个全连接层
        self.fc1 = layers.Dense(16, activation=activation)
        self.fc2 = layers.Dense(32, activation=activation)
        self.fc3 = layers.Dense(64, activation=activation)
        self.fc4 = layers.Dense(32, activation=activation)
        self.fc5 = layers.Dense(32, activation=activation)
        self.fc6 = layers.Dense(16, activation=activation)
        self.fc7 = layers.Dense(8, activation=activation)
        # self.fc8 = layers.Dense(16, activation=activation)
        self.fc8 = layers.Dense(2)
    def call(self, inputs, training=None, mask=None):
        # 依次通过 N 个全连接层
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        # x = self.fc9(x)
        return x

    def loss_BC(self, x, y):
        loss_u = tf.reduce_mean(tf.square(y - self.call(x)))
        return loss_u

    def loss_PDE(self, x_to_train_f):
        g = tf.Variable(x_to_train_f, dtype='float64', trainable=False)
        x0 = g[:, 2:4]
        t = g[:, 4:5]
        Tf = 300
        caf = 10
        Tc = 310
        ca = g[:, 2:3]
        T = g[:, 3:4]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            t = t * K_t
            g = tf.stack([x0[:, 0], x0[:, 1], t[:, 0]], axis=1)
            z = self.call(g)
            z_ca_inverse = z[:, 0:1] * K_ca + ca_min
            z_T_inverse = z[:, 1:2] * K_T + T_min
            ca_t = tape.gradient(z_ca_inverse, t)
            T_t = tape.gradient(z_T_inverse, t)

        del tape

        # ca_t_inverse = ca_t * (K_t/K_ca)
        # T_t_inverse = T_t * (K_t/K_T)
        # caf_inverse = scaler_caf.inverse_transform(caf)
        # Tc_inverse = scaler_Tc.inverse_transform(Tc)
        ca_inverse = scaler_ca.inverse_transform(ca)
        T_inverse = scaler_T.inverse_transform(T)

        r = 34930800 * tf.exp(-11843 / (1.985875 * T_inverse)) * ca_inverse
        f_ca = (caf - ca_inverse) - r - ca_t
        f_T = (Tf - T_inverse) - (-5960 / 500) * r - (150 / 500) * (T_inverse - Tc) - T_t

        loss_f = tf.reduce_mean(tf.square(tf.stack([f_ca, f_T], axis=1)))
        loss_f = tf.cast(loss_f, dtype=tf.float32)

        return loss_f

    def loss(self, x, y, g):
        loss_u = self.loss_BC(x, y)
        loss_f = self.loss_PDE(g)
        # loss_f = loss_u

        loss = loss_u + lambda1 * loss_f

        return loss, loss_u, loss_f


model = Network()  # 创建网络类实例
# 通过 build 函数完成内部张量的创建，其中None为任意的batch数量，9为输入特征长度
model.build(input_shape=(None, 3))
model.summary()  # 打印网络信息

N_u = 8000  # Total number of data points for 'u'
N_f = 1000  # Total number of collocation points
lambda1 = 0.0001

xf_train, x0_train, x1_train = trainingdata(N_u, N_f)  # Training data

optimizer = tf.keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.004)  # 创建优化器，指定学习率

for epoch in range(1000):  # N 个 Epoch
    # 梯度记录器
    with tf.GradientTape() as tape:
        loss = model.loss(x0_train, x1_train, xf_train)  # 计算 loss
    print(epoch, loss[0].numpy(), loss[1].numpy(), loss[2].numpy())
    # 计算梯度，并更新
    grads = tape.gradient(loss[0], model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


sim_length = 9000
x = np.empty([sim_length, 2])
u = np.empty([sim_length, 2])
t = np.empty([sim_length, 1])
x_input = np.empty([sim_length, 3])
x[0, :] = data_minMax[0, 2:4]
for i in range(sim_length-1):
    # u[i, :] = data_minMax[i, 0:2]
    t[i, :] = data_minMax[i, 4]
    x_input[i, :] = np.hstack([x[i, :], t[i, :]])
    x_pred = model.call(x_input[i:i+1, :])
    x[i+1, :] = x_pred

compare = np.empty([sim_length, 4])
compare[:, 0:2] = x[:, 0:2]
compare[:, 2:4] = data_minMax[0:sim_length, 2:4]
# mse = mean_squared_error(compare[:, 0], compare[:, 1])
# print(mse)
plt.plot(compare)
plt.title(lambda1)
plt.show()