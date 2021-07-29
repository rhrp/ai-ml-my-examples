#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:32:32 2021
@author: rhp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
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
def plot_hist_dataset(dataset,bins,p_imgs_per_col,show):
    #figsize=(30,15)
    #layout=(10,2)
    total_rows=dataset.index.size
    total_cols=dataset.columns.size
    print('total_rows=',total_rows)
    print('total_cols=',total_cols)
    imgs_lines=int(total_cols/p_imgs_per_col)+1
    hist = dataset.hist(bins=bins,figsize=(20,20),layout=(imgs_lines,p_imgs_per_col))
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
    Performs the data cleaning of a dataset based on the structure of the Pima Indian Diabetes.
    Do not use with another dataset!
"""
def clean_invalid_data_pima_indian_diabetes_dataset(dataset):
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

"""
    Performs the data cleaning of a dataset based on the limit fences of IQR
    p_dataset - Pandas dataset
    p_col_name - Name of the colunm
    p_verbose - True/False
"""
def clean_iqr_outliers(p_dataset,p_col_name,p_verbose):
    q1=np.quantile(p_dataset[p_col_name],0.25)  # Q1
    q3=np.quantile(p_dataset[p_col_name],0.75)  # Q3
    iqr=q3-q1
    fence1=q1-iqr*3.0
    fence2=q1-iqr*1.5
    fence3=q3+iqr*1.5
    fence4=q3+iqr*3.0

    is_up_fence2=p_dataset[p_col_name]>=fence2
    is_dw_fence3=p_dataset[p_col_name]<=fence3
    dataset_cleaned=p_dataset[is_up_fence2 & is_dw_fence3]
    
    # reindex after eliminate rows
    dataset_cleaned.reset_index(drop=True, inplace=True)
    
    if p_verbose:
        print('Eliminating the outliers of %s  irq=%.3f  [%.3f ... %.3f]'%(p_col_name,iqr,fence2,fence3))
        print("Before: ",len(p_dataset))
        print("After data cleaning: ",len(dataset_cleaned))

    return dataset_cleaned

"""
This method performs a graphical analyses of a feature in a Pandas dataset
"""
def analysis_graph(p_dataset,p_col_name_feature,p_col_name_output,p_title):
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(p_title, fontsize=16)
    
    plt.subplot(1,4,1)
    p_dataset[p_col_name_feature].plot.hist(title='Pandas Histogram',alpha=0.5,bins=100)

    #plt.subplot(1,3,2)
    #sns.histplot(p_col_data.to_numpy(),bins=100) # as you see most of values are scattered between the values 40 and 60 as it is obvious that we have chosen a normal distrubition hving an average value of 50 and standart deviaton of 10
    #plt.title('sns.histplot')
    
    plt.subplot(1,4,2)
    plt.plot(p_dataset[p_col_name_output], p_dataset[p_col_name_feature],'.')
    plt.title('Distibution per '+p_col_name_output)
    
    sns.set_theme(style="whitegrid")
    plt.subplot(1,4,3)
    ax = sns.violinplot(x=p_col_name_output,y=p_col_name_feature,data=p_dataset)
    ax.set_title('Violin Plot')
    
    plt.subplot(1,4,4)
    sns.boxplot(p_dataset[p_col_name_feature])
    plt.title('boxplot')
    
    plt.show()
    return fig

def analysis_graph_dataset(p_dataset,p_col_name_output,p_title):
    fig = plt.figure(figsize=(40, 40))
    fig.suptitle(p_title, fontsize=16)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    lines=p_dataset.columns.size-1
    position=0
    for c in p_dataset.columns:
        #print(c)
        if c not in [p_col_name_output]:
            position=position+1
            plt.subplot(lines,4,position)
            p_dataset[c].plot.hist(title='Pandas Histogram of '+c,alpha=0.5,bins=100)

            #plt.subplot(1,3,2)
            #sns.histplot(p_col_data.to_numpy(),bins=100) # as you see most of values are scattered between the values 40 and 60 as it is obvious that we have chosen a normal distrubition hving an average value of 50 and standart deviaton of 10
            #plt.title('sns.histplot')
            
            position=position+1
            plt.subplot(lines,4,position)
            plt.plot(p_dataset[p_col_name_output], p_dataset[c],'.')
            plt.title('Distibution of '+c+' per '+p_col_name_output)

            position=position+1
            sns.set_theme(style="whitegrid")
            plt.subplot(lines,4,position)
            ax = sns.violinplot(x=p_col_name_output,y=c,data=p_dataset)
            ax.set_title('Violin Plot '+c)

            position=position+1
            plt.subplot(lines,4,position)
            sns.boxplot(p_dataset[c])
            plt.title('Boxplot '+c)

    plt.show()
    return fig