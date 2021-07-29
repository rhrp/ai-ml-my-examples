#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:40:49 2021

@author: rhp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

param_components=1
param_show_digit=None   # None, 0..9

ds_mnist = pd.read_csv('/tmp/mnist_train.csv')

# get the labels
labels = ds_mnist['label']

# Drop the label feature in order to get only the data in the dataset
data = ds_mnist.drop("label",axis=1)

# Data structure
print(data.head())
print('Data: ',data.shape)
print('Labels: ',labels.shape)


# plot the first 4 images of one of the digits
n=0
if param_show_digit in range(0,10):
    for i in range(0,len(labels)):
        if labels[i]==param_show_digit and n<3:
            plt.figure(figsize=(7,7))
            grid_data = data.iloc[i].to_numpy().reshape(28,28)  # reshape from 1d to 2d pixel array
            plt.imshow(grid_data, interpolation = "none", cmap = "gray")
            plt.show()
            n=n+1


# Data-preprocessing: Standardizing the data
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
print('Standardized Data: ',standardized_data.shape)


lda = LinearDiscriminantAnalysis(n_components=param_components)
data_reduced = lda.fit(standardized_data, labels).transform(standardized_data)


# pca_reduced will contain the 2-d projects of simple data
print("shape of lda_data.shape = ", data_reduced.shape)

if param_components==3:
    fig = plt.figure(1, figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
for digit in [0,1,2,3,4,5,6,7,8,9]:
    print('Digit-',digit)
    idx_tuple=np.where(labels==digit)
    idx=idx_tuple[0]
    # get the idx array from the tuple of results
    print('idx: ',idx.shape)
    # Get the data of the selected digit
    lda_data_digit_arr=np.asarray(data_reduced[idx])
    print('lda_data_digit_arr: ',lda_data_digit_arr.shape)
    # Convert  N x 2 to 2 x N
    lda_data_digit=lda_data_digit_arr.T
    print('lda_data_digit: ',lda_data_digit.shape)
    # Plot the digit's data
    if param_components==1:
        #This array enables that each digit being ploted in a colunm
        tmp=np.zeros(len(lda_data_digit_arr))+digit
        plt.plot(tmp,lda_data_digit[0],'.',label=str(digit))
    elif param_components==2:
        plt.plot(lda_data_digit[0],lda_data_digit[1],'.',label=str(digit))
    elif param_components==3:
        ax.scatter(lda_data_digit[0],lda_data_digit[1],lda_data_digit[2],c=None,cmap=plt.cm.Set1, edgecolor='k', s=40,label=str(digit))
    else:
        print('Error: invalid param_components')

if param_components==3:
    ax.legend()
    ax.set_title('LDA of MNIST  3 components')
else:
    plt.legend()
    plt.title('LDA of MNIST '+str(param_components)+' components')

plt.show()