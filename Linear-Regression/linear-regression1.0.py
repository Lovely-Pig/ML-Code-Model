import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# m表示样本的数目，n表示特征的数目
X_train = <np.array>    # X_train.shape = (m, n + 1)
y_train = <np.array>    # y_train.shape = (n, 1)

# >>> X_train
# np.array([[1, x, x, x, x, ],
#           [1, x, x, x, x, ],
#                  ...
#           [1, x, x, x, x, ]])

# Hypothesis Funcyion
def hypothesis(X_data, w):
    y_hat = np.matmul(X_data, w)
    return y_hat

# Loss Function
def loss(X_data, y_hat, y_data, m):
    error = (1 / (2 * m)) * np.sum(np.square(y_hat - y_data))
    return error

# grad of Loss Function
def grad(X_data, y_hat, y_data, m):
    w_grad = (1 / m) * np.matmul(X_data.T, y_hat - y_data)
    return w_grad

# 初始化
learning_rate = <float>  # 笔者建议0.0001、0.001、0.01、0.1、...
iter_time = <int>    # 笔者建议1000、2000、5000、10000、...
w = np.random.randn((n + 1, 1))

# 进行gradient descent
for i in range(iter_time):
    y_hat = hypothesis(X_train, w)    # y_hat.shape = (m, 1)
    training_error = loss(X_train, y_hat, y_train, m)
    w_grad = grad(X_train, y_hat, y_data, m)    # w_grad.shape = (n + 1, 1)
    w = w - learning_rate * w_grad
