#!/usr/bin/env python
"""
Demonstrates the cleaning and normalization of data.

The Indian Diabetes dataset has 768 rows.
However, many of them have zeros as values, which are wrong values, e.g.
in the case of BloodPressure. Nobody has such blood pressure.
ToDo: Instead of dropout these records, replace the
zeros by the mean value of the feature

The tests demonstrate that normalization improves the learning, avoiding
erratic (up/down) evolutions of loss and accuracy. 

On the other hand, applying the standarization, the model stop working :-(
ToDO: Why?

Thanks to  Jason Brownlee:
https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
"""
__author__ = "Rui Humberto Pereira"
__copyright__ = "Copyright 2021, RHP"
__credits__ = ["Me","Jason Brownlee"]

__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Rui Humberto Pereira"
__email__ = "rui.humberto.pereira@gmail.com"
__status__ = "Development"

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
import rhplib.datasets_pandas
import rhplib.plot_functions

        
param_epochs=50
param_learningrate=0.001
param_do_clean=True
param_do_scaling=1  # 0-None, 1-Normalizatio and 2-Standardization

# load the dataset using numpy
#from numpy import loadtxt
#dataset = loadtxt('../datasets/PimaIndiansDiabetesDatabase.csv', skiprows=1, delimiter=',')
# split into input (X) and output (y) variables
#X = dataset[:,0:8]
#y = dataset[:,8]

# load Pima dataset
dataset = pd.read_csv('../datasets/PimaIndiansDiabetesDatabase.csv',delimiter=',',header=0)
#print(dataset.head())
rhplib.datasets_pandas.plot_hist_dataset(dataset,100)

if param_do_clean:
    #Clean the data
    dataset=rhplib.datasets_pandas.clean_pima_indian_diabetes_dataset(dataset)
    #print(dataset.head())
    rhplib.datasets_pandas.plot_hist_dataset(dataset,100)

if param_do_scaling==1:
    #Normalize the data
    dataset=rhplib.datasets_pandas.normalize(dataset);
    #tmp_dataset=tmp_dataset.drop(0, axis='index')
    dataset.to_csv("../datasets/tmp.csv",index=False)
    #print(dataset.head())
    rhplib.datasets_pandas.plot_hist_dataset(dataset,100)
elif param_do_scaling==2:
    #Standardization the data
    dataset=rhplib.datasets_pandas.standardization(dataset);
    #tmp_dataset=tmp_dataset.drop(0, axis='index')
    dataset.to_csv("../datasets/tmp.csv",index=False)
    #print(dataset.head())
    rhplib.datasets_pandas.plot_hist_dataset(dataset,100)
else:
    print('No data scaling!')
#
dataset_train, dataset_validation = train_test_split(dataset, test_size=0.1)
print('Dataset Train:',len(dataset_train),'\nDataset validation:',len(dataset_validation))

#rhplib.datasets_pandas.plot_hist_col(dataset.Pregnancies,20)
#rhplib.datasets_pandas.plot_hist_col(dataset.Age,120)
#rhplib.datasets_pandas.plot_hist_col(dataset.Glucose,120)
#rhplib.datasets_pandas.plot_hist_col(dataset.BloodPressure,120)


# split into input (X) and output (y) variables
X = dataset_train.values[:,0:8]
y = dataset_train.values[:,8]


X_validation = dataset_validation.values[:,0:8]
y_validation = dataset_validation.values[:,8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Optimizer
my_opt = Adam(learning_rate=param_learningrate)
# compile the keras model (loss:binary_crossentropy,mse)
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

# fit the keras model on the dataset
train_history = model.fit(X, y, epochs=param_epochs, batch_size=10,validation_data=(X_validation,y_validation))

#plot the model performance
train_loss = train_history.history['loss']
train_accuracy = train_history.history['accuracy']
test_loss = train_history.history['val_loss']
test_accuracy = train_history.history['val_accuracy']
rhplib.plot_functions.plot_train_and_test_loss_accuracy(train_loss, train_accuracy, test_loss, test_accuracy)



# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))




