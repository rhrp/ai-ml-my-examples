# -*- coding: utf-8 -*-
import numpy

import numpy as np

# Sigmoid activation function
def sigmoid(z):
    y = 1 / (1 + np.exp(-1*z))
    return y

# ReLu activation function
def ReLU(z):
    if z > 0:
        return z
    else:
        return 0
    
#activation
def activation(z):
    v=sigmoid(z)
    y=1 if v>0.5 else 0  
    #print(z,'   ',v,'    ',y)
    return y

#
#Perceptron - Requires the weights and biases, because it cannot learn
#
def perceptron(X,W,bias):
    z=0
    for i in range(0,len(X)):
        z=z+X[i]*W[i]
    z=z+bias
    y_pred=activation(z)
    #print('z=',z,'   (',numpy.dot(W,X)+bias,')   Y=',y_pred)
    return y_pred

#Dataset
ds_x=[[0,0],[0,1],[1,0],[1,1]]


#And
f_name="And"
W=[1,1]
Bias=-1.5
for x in ds_x:
    print(x[0],f_name,x[1],' = ',perceptron(x,W,Bias))

#Or
f_name="Or"
W=[1,1]
Bias=-0.5
for x in ds_x:
    print(x[0],f_name,x[1],' = ',perceptron(x,W,Bias))
    
#Not
ds_x=[[0],[1]]
f_name="Not"
W=[-1]
Bias=0.5
for x in ds_x:
    print(x[0],f_name,' = ',perceptron(x,W,Bias))
    
    
def p_and(X):
    W=[1,1]
    Bias=-1.5
    return perceptron(X,W,Bias)
def p_or(X):
    W=[1,1]
    Bias=-0.5
    return perceptron(X,W,Bias)
def p_not(X):
    W=[-1]
    Bias=0.5
    return perceptron(X,W,Bias)
def p_xor(X):
    v_and=p_and(X)
    v_not_and=p_not([v_and])
    v_or=p_or(X)
    return p_and([v_not_and,v_or])

#XOR
#Dataset
ds_x=[[0,0],[0,1],[1,0],[1,1]]
f_name="Xor"
for x in ds_x:
    print(x[0],f_name,x[1],' = ',p_xor(x))