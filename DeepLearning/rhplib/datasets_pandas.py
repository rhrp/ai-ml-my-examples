#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:32:32 2021
@author: rhp
"""
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_2D_column(col1,col2):
    x = col1
    y = col2
    plt.scatter(x, y)
    plt.show()
    
def plot_3D_column(col1,col2,col3):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(col1,col2,col3, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three Pandas")
    ax.set_xlabel(col1.name)
    #ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel(col2.name)
    #ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel(col3.name)
    #ax.w_zaxis.set_ticklabels([])
    plt.show()

"""
 Plots the histogram of all columns of the Pandas dataset
"""
def plot_hist_dataset(dataset,bins,show):
    hist = dataset.hist(bins=bins)
    if show:
        plt.show()
    
"""
 Plots the histogram of a column of the Pandas dataset. The X distribution can be provided in the bins argument
"""
def plot_hist_col(col,bins,show):
    hist = col.hist(bins=bins)
    hist.set_title(col.name)
    if show:
        plt.show()
    
"""
    Performs the normanization of the dataset based on MinMaxScaler
    Using this method you lose the scaler that wil may you inverse
    the transformation of predicted values
    # inverse transform
    inverse = scaler.inverse_transform(normalized)
"""
def normalize(df):
    data = df.values #returns a numpy array
    names=df.columns
        
    #create scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    #TODO: Which is the diference form keras's nornalization
    #import tensorflow as tf
    #data_scaled=tf.keras.utils.normalize(data, axis=1)

    df_normalized = pd.DataFrame(data_scaled, columns=names)    
    print('\nBefore Normalization(',len(df),'):\n',df);
    print('\nAfter Normalization(',len(df_normalized),'):\n',df_normalized)
    return df_normalized

"""
    Performs the Standardization of the dataset based on StandardScaler
    Using this method you lose the scaler that wil may you inverse
    the transformation of predicted values
    # inverse transform
    inverse = scaler.inverse_transform(standardized)
"""    
def standardization(df):
    data = df.values #returns a numpy array
    names=df.columns
    
    # create scaler
    scaler = preprocessing.StandardScaler()
    # fit and transform in one step
    data_scaled = scaler.fit_transform(data)
    df_standardized = pd.DataFrame(data_scaled, columns=names)    
    print('\nBefore standardized(',len(df),'):\n',df);
    print('\nAfter standardized(',len(df_standardized),'):\n',df_standardized)
    return df_standardized

"""
    Performs the data cleaning of a dataset based on the 
    structure of the Pima Indian Diabetes.
    Do not use with another dataset!
"""
def clean_pima_indian_diabetes_dataset(dataset):
    is_Glucose_not_0=dataset.Glucose>60
    is_BMI_not_0=dataset.BMI>0.1
    is_Insulin_not_0=dataset.Insulin>0.1
    is_SkinThickness_not_0=dataset.SkinThickness>0.1
    is_BloodPressure_not_0=dataset.BloodPressure>10
    dataset_cleaned=dataset[is_Glucose_not_0 & is_BMI_not_0 & is_Insulin_not_0 & is_SkinThickness_not_0 & is_BloodPressure_not_0]
    # reindex after eliminate rows
    dataset_cleaned.reset_index(drop=True, inplace=True)
    print("Before: ",len(dataset))
    print("After data cleaning: ",len(dataset_cleaned))
    return dataset_cleaned