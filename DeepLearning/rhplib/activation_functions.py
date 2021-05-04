import numpy as np

# Sigmoid activation function
def sigmoid(y):
    z = 1 / (1 + np.exp(-y))
    return z

# ReLu activation function
def ReLU(y):
    if y > 0:
        return y
    else:
        return 0