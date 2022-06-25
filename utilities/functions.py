import numpy as np


def ReLU(tensor_1d):
    return np.maximum(0, tensor_1d)

def dReLU(tensor_1d):
    return np.where(tensor_1d <= 0, 0, 1)

activations = {"ReLU": ReLU,
               "dReLU": dReLU}

def MSE(output, expected):
    return np.power(np.subtract(expected, output), 2)

def dMSE(output, expected):
    return 2*np.subtract(expected, output)

cost_functions = {"MSE": MSE,
                  "dMSE": dMSE}
