#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 12:30:05 2021

@author: rhp
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow import keras
import math


X_train=np.array(range(1,1000))
y_train_without_noise=np.array(range(1,1000))
y_train=np.zeros(len(X_train))
for i in range(0,len(X_train)):
    y_train[i]=y_train_without_noise[i]+150*math.sin(np.pi*(i/len(X_train)))-75


#Overfits
PARAM_NODES=32
PARAM_EPOCHS=100
PARAM_ADD_DROPOUT=False

#Generalizes, dont Overfits
PARAM_NODES=32
PARAM_EPOCHS=100
PARAM_ADD_DROPOUT=True

#Generalizes, dont Overfits
#PARAM_NODES=3
#PARAM_EPOCHS=50
#PARAM_ADD_DROPOUT=False

# define the keras model
model = Sequential()
model.add(Dense(PARAM_NODES, activation='relu',input_shape=[1]))
if PARAM_ADD_DROPOUT:
    model.add(Dropout(0.2))
    print("Added dropout layer")
model.add(Dense(PARAM_NODES, activation='relu'))
if PARAM_ADD_DROPOUT:
    model.add(Dropout(0.2))
    print("Added dropout layer")
model.add(Dense(1,activation='linear'))

#
#my_opt = keras.optimizers.SGD()
my_opt = keras.optimizers.RMSprop(0.0099)
my_opt = keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-05, amsgrad=False,
    name='Adam')
model.compile(loss='mse', optimizer=my_opt, metrics=['mse','mae'])

#
train_history=model.fit(X_train,y_train, epochs=PARAM_EPOCHS,batch_size=1,validation_split=0.1)
train_loss = train_history.history['loss']
val_loss = train_history.history['val_loss']


figure = plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['Train loss', 'Validation loss'])
plt.show()


#
y_pred=model.predict(X_train)
plt.plot(X_train,y_train,'.',color='blue')
plt.plot(X_train,y_train_without_noise,color='green')
plt.plot(X_train,y_pred,color='red')
plt.show()

# evaluate the keras model
#loss, accuracy = model.evaluate(X, Y)
#figure = plt.figure()
#plt.plot(loss)
#plt.plot(accuracy)
#plt.legend(['Train loss', 'Train accuracy'])
#plt.show()