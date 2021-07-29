#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:40:49 2021

@author: rhp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

param_components=2
param_n_points=10000
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


from sklearn.manifold import TSNE

# Picking the top 1000 points as TSNE takes a lot of time for 15K points
data_part = standardized_data[0:param_n_points,:]
labels_part = labels[0:param_n_points]

model = TSNE(n_components=param_components, random_state=0,perplexity=50)
# configuring the parameteres
# the number of components = 2 or 3
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data_part)


# pca_reduced will contain the 2-d projects of simple data
print("shape of tsne_data.shape = ", tsne_data.shape)

if param_components==3:
    fig = plt.figure(1, figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
for digit in range(0,10):
    print(digit)
    idx_tuple=np.where(labels_part==digit)
    idx=idx_tuple[0]
    # get the idx array from the tuple of results
    print('idx: ',idx.shape)
    # Get the data of the selected digit
    tsne_data_digit_arr=np.asarray(tsne_data[idx])
    print('tsne_data_digit_arr: ',tsne_data_digit_arr.shape)
    # Convert  N x 2 to 2 x N
    tsne_data_digit=tsne_data_digit_arr.T
    print('tsne_data_digit: ',tsne_data_digit.shape)
    # Plot the digit's data
    if param_components==2:
        plt.plot(tsne_data_digit[0],tsne_data_digit[1],'.',label=str(digit))
    elif param_components==3:
        ax.scatter(tsne_data_digit[0],tsne_data_digit[1],tsne_data_digit[2],c=None,cmap=plt.cm.Set1, edgecolor='k', s=40,label=str(digit))
    else:
        print('Error: invalid param_components')

if param_components==3:
    ax.legend()
    ax.set_title('TSNE of MNIST  3 components')
else:
    plt.legend()
    plt.title('TSNE of MNIST '+str(param_components)+' components')

plt.show()