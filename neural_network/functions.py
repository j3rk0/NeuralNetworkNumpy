import math
import numpy as np


def tanh(Z):
    tanh.derivative = tanh_derivative
    return np.tanh(Z)


def tanh_derivative(Z):
    return 1.0 - np.tanh(Z) ** 2


def leaky_relu(Z):
    leaky_relu.derivative = leaky_relu_derivative
    return np.where(Z > 0, Z, Z * 0.01)


def leaky_relu_derivative(Z):
    ret = np.ones(Z.shape)
    ret[Z < 0] = .01
    return ret


# ----------------------------------------------------------

def relu(Z):
    relu.derivative = relu_derivative
    return np.maximum(Z, 0)


def sigmoid(Z):
    sigmoid.derivative = sigmoid_derivative
    return 1 / (1 + np.power(np.e, -Z))


def relu_derivative(Z):
    dZ = np.zeros(Z.shape)
    dZ[Z > 0] = 1
    return dZ


def sigmoid_derivative(Z):
    f = 1 / (1 + np.exp(-Z))
    return f * (1 - f)


def softMaxCe(x):
    softMaxCe.derivative = derivSoftMaxCe
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def derivSoftMaxCe(x):
    return 1


# %%


def crossEntropy(y, t):
    crossEntropy.derivative = derivCrossEntropy
    return np.sum(y * np.log(t))


def crossEntropySM(y, t):
    crossEntropySM.derivative = derivCrossEntropySM
    a = np.log(y + 1e-10)  # evitiamo che ci siano zeri
    return np.sum(t * a)


def sumOfSquare(y, t):
    sumOfSquare.derivative = derivSumOfSquare
    return np.sum(np.square(y - t))


def derivCrossEntropySM(y, t):
    return y - t


def derivCrossEntropy(y, t):
    return -y / t + (1 - t) / (1 - y)


def derivSumOfSquare(y, t):
    return y - t
