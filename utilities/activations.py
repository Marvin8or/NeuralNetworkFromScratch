import numpy as np


def ReLU(tensor_1d):
    return np.maximum(0, tensor_1d)

functions = {"ReLU": ReLU}
