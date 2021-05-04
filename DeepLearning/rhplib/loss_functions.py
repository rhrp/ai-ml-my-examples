from tensorflow import losses
import numpy as np

# Mean Square Error, Quadratic loss
def mse(Y,P):
    sum = 0
    for i in range(0,len(Y)):
        sum=sum+(Y[i]-P[i])**2
    return sum/len(Y)

#Dont work :-(
def mse_(Y,P):
    return np.mean(np.abs(Y - P))

def mse_tk(Y,P):
    loss = losses.mean_absolute_error(Y, P)
    return loss.numpy()

# Mean Absolute Error (MAE)
def mae(Y,P):
    sum = 0
    for i in range(0,len(Y)):
        sum=sum+np.abs((Y[i]-P[i]))
    return sum/len(Y)

# BinaryCrossentropy
def BinaryCrossentropy(Y,P):
    bce = losses.BinaryCrossentropy()
    return bce(Y, P).numpy()
