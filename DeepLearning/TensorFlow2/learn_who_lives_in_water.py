#
# This Dataset too small :-(
#
#
import datetime
import os

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from keras.layers import Embedding
from tensorflow import keras
from sklearn.model_selection import train_test_split
from rhplib import plot_functions as rhpplot

param_embeddings_on=True
param_epochs=20
param_learningrate=0.01

#load data from a CSV
csv_original = pd.read_csv('../datasets/animals.csv', header=0, sep=';')
csv = csv_original.copy()
print(csv)
csv_train, csv_valid = train_test_split(csv, test_size=0.1)

#csv_valid=csv_train

#print('\nEntire dataset:\n ',csv)
print('\nTrain dataset:\n',csv_train)
print('\nValidate dataset:\n',csv_valid)

csv_train=csv_train.drop('name',1)
#get the N samples of 4 features == array N x 4  (filter columns  1 to the end)
train_data = csv_train.values[:,[0,1,2,3]]
#get the targets for each sample == array 1 x N (#filter the first column)
train_target = csv_train.values[:,4]

csv_valid=csv_valid.drop('name',1)
valid_data=csv_valid.values[:,[0,1,2,3]]
valid_target=csv_valid.values[:,4]

print('\nTrain:\n',csv_train)
print('\nTrain Data:  ',train_data)
print('\nTrain target:',train_target)

print('\nValidate:\n',csv_valid)
print('\nValidate Data:  ',valid_data)
print('\nValidate target:',valid_target)

labels = csv_original.values[:,0]
#labels = csv_original.values[:,5]
print (labels)


log_dir = "/tmp/logs/fit/animals_" + datetime.datetime.now().strftime("%Y%m%d-%Hh%M.%S")
metadata_path=log_dir+'/metadata.tsv'

# define the keras model
model = Sequential()
if param_embeddings_on:
    embed_size=4
    vocab_size=len(labels)
    model.add(Embedding(input_dim=vocab_size,output_dim=embed_size,input_length=4, input_shape=[4], name='embedding'))
    model.add(keras.layers.Flatten())
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10, batch_size=100,write_graph=True,write_grads=True, write_images=True,embeddings_freq=1, embeddings_metadata={'embedding': metadata_path})
else:
    model.add(InputLayer(4))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.add(Dense(100,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#my_opt = keras.optimizers.SGD(learning_rate=param_learningrate,momentum=0)
my_opt = keras.optimizers.Adam(learning_rate=param_learningrate)
# Loss: 'binary_crossentropy','mse'
# Metrics:  'accuracy','binary_crossentropy','mse'
model.compile(loss='mse', optimizer=my_opt, metrics=['accuracy'])

# fit the keras model on the dataset
train_history = model.fit(train_data,train_target, epochs=param_epochs, batch_size=1, verbose=1, validation_data=(valid_data, valid_target),callbacks=[tensorboard_callback])
train_loss = train_history.history['loss']
train_accuracy = train_history.history['accuracy']
test_loss = train_history.history['val_loss']
test_accuracy = train_history.history['val_accuracy']

rhpplot.plot_train_and_test_loss_accuracy(train_loss, train_accuracy, test_loss, test_accuracy)


print(model.summary())

#Save the metadata for TensorBoard
with open(metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(labels):
        print(index,'   ->   ',label)
        meta.write('{}\t{}\n'.format(index, label))
   
# Test the model
has_legs=0
can_swim=1
has_gills=1
breathes=0
p = model.predict([[has_legs,can_swim,has_gills,breathes]])
print('\n\n--------------------------------------')
print('\nhas_legs=%d\ncan_swim=%d\nhas_gills=%d\nbreathes=%d'%(has_legs,can_swim,has_gills,breathes))
print('\nlives in the water: ','yes' if p>0.5 else 'no','  p=',p)
print('\n--------------------------------------')
