import numpy as np
import math
from matplotlib import pyplot as plt


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        #         print(pt[0],pt[1])
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_eazy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()


def show_resultfinal(x, y, pred_y, learncurve, ty):
    marklabel = "cross entrpy"
    if ty:
        marklabel = "Mean square error"
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 3, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 3, 3)
    plt.title('Learning Curve', fontsize=18)
    plt.plot(learncurve)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def accu(pred, actual):
    k = 0.0
    if len(pred) != len(actual):
        print("wrong")
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            k += 1
    return k / len(pred) * 100


def printimportant(i, lendata):
    loss = "cross entrpy"
    datatype = "generate_linear(n=100)"
    if i:
        loss = "Mean square error"
    if lendata == 21:
        datatype = "generate_XOR_eazy()"
    print("loss_function:{}, Data:{}".format(loss, datatype))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def de_sigmoid(x):
    return np.multiply(x, 1.0-x)
def relu(x):
    return np.maximum(x,0)
def de_relu(x):
    return np.minimum(np.maximum(x,0),1)

def act_func(z, func_type):
    if (func_type == "relu"):
        return relu(z)
    elif (func_type == "sigmoid"):
        return sigmoid(z)
    else:
        return z


def de_act_func(z, func_type):
    if (func_type == "relu"):
        return de_relu(z)
    elif (func_type == "sigmoid"):
        return de_sigmoid(z)
    else:
        return z


class network:
    def __init__(self, hidden_size, epoch, lr, costfunction, h1_act_func, h2_act_func, out_act_func):

        self.h1_act_func = h1_act_func
        self.h2_act_func = h2_act_func
        self.out_act_func = out_act_func

        self.costf = costfunction
        self.epoch = epoch
        self.lr = lr
        self.EPS = 1e-9
        self.learncurve = []

        self.x = np.zeros((2, 1))
        self.w = [np.random.rand(hidden_size[0], 2), np.random.rand(hidden_size[1], hidden_size[0]),
                  np.random.rand(1, hidden_size[1])]
        self.z = [np.zeros((hidden_size[0], 1)), np.zeros((hidden_size[1], 1)), np.zeros((1, 1))]
        self.a = [np.zeros((hidden_size[0], 1)), np.zeros((hidden_size[1], 1)), np.zeros((1, 1))]

    def forward(self, inputt):
        self.x = inputt
        self.z[0] = self.w[0] @ self.x
        self.a[0] = act_func(self.z[0], self.h1_act_func)
        self.z[1] = self.w[1] @ self.a[0]
        self.a[1] = act_func(self.z[1], self.h2_act_func)
        self.z[2] = self.w[2] @ self.a[1]
        self.a[2] = act_func(self.z[2], self.out_act_func)

        return self.a[2][0][0]

    def backward(self, actual, pred):
        if self.costf:
            self.cost = np.vdot((actual - pred), (actual - pred))
            pCa2 = -2 * (actual - pred)
        #             print("mse")
        else:
            self.cost = -(actual * math.log(pred + 1e-9) + (1 - actual) * math.log(1 - pred + 1e-9))
            pCa2 = -(actual / (pred + self.EPS) - (1 - actual) / (1 - pred + self.EPS))
        #             print("ce")

        pa2z2 = de_act_func(self.a[2], self.out_act_func)
        pCz2 = pCa2 * pa2z2
        pCw2 = self.a[1] * pCz2
        pCw2 = pCw2.T

        pz1w1 = self.a[0]
        pCa1 = pCz2.T @ self.w[2]
        pa1z1 = de_act_func(self.a[1], self.h2_act_func)
        n = pCa1.shape[1]
        repCa1 = []
        for j in range(n):
            repCa1.append(pCa1[0][j])
            if j != n - 1:
                for k in range(n):
                    repCa1.append(0)
        repCa1 = np.array(repCa1).reshape(n, n)
        pCz1 = repCa1 @ pa1z1
        pCw1 = pCz1 @ pz1w1.T

        pz0w0 = self.x
        pCa0 = pCz1.T @ self.w[1]
        pa0z0 = de_act_func(self.a[0], self.h1_act_func)
        n = pCa0.shape[1]
        repCa0 = []
        for j in range(n):
            repCa0.append(pCa0[0][j])
            if j != n - 1:
                for k in range(n):
                    repCa0.append(0)
        repCa0 = np.array(repCa0).reshape(n, n)
        pCz0 = repCa0 @ pa0z0
        pCw0 = pCz0 @ pz0w0.T

        self.w[0] -= self.lr * pCw0
        self.w[1] -= self.lr * pCw1
        self.w[2] -= self.lr * pCw2


####################################################################

data,label = generate_XOR_eazy()
# data, label = generate_linear(n=100)

# 1:Mean square error, 0:corss entropy
costfunction = 1
hidden_size = (8, 8)
learning_rate = 0.05
epoch = 2000
print_step = 500
comp = 0

#          hidden_size, epoch, lr,costfunction, h1_act_func, h2_act_func, out_act_func
gg = network(hidden_size, epoch, learning_rate, costfunction, "", "sigmoid", "sigmoid")

printimportant(costfunction, len(data))
for ep in range(gg.epoch):

    predylist = []
    lecurve = 0
    for idx in range(len(data)):
        inputt = data[idx].reshape(2, 1)
        actual = label[idx]

        pred = gg.forward(inputt)
        gg.backward(actual, pred)
        predylist.append([pred])
        lecurve += gg.cost
    gg.learncurve.append(lecurve / len(data))

    if ep % print_step == 0:
        predy = np.array(predylist)
        t = accu(np.round(predy), label)
        print("epoch {} loss : {}  accuracy : {:.2f}%".format(ep, gg.learncurve[-1], t))
        print(predy)
        if (comp != t):
            comp = t
            show_result(data, label, np.round(predy))

predy = np.array(predylist)
print("epoch {} loss : {}  accuracy : {:.2f}%".format(ep + 1, gg.learncurve[-1], accu(np.round(predy), label)))
print(predy)
show_resultfinal(data, label, np.round(predy), gg.learncurve, costfunction)
s = gg.learncurve

# predylist = []






