#
# Logical AND function
#
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import datetime
import rhplib.plot_functions as rhpplot
import rhplib.datasets_bin as rhpds

#Mode 1 - Train the model saving the data using a tensorboard callback
#Mode 2 - Train the model ploting the loss and accuracy trajectory
param_test_mode=2
param_epochs=150
param_learningrate=1

# split into input (X) and output (y) variables
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]

rhpds.plotBinaryDataSet(X,y)

# define the keras model
model = Sequential()
model.add(InputLayer(2))
model.add(Dense(1, activation='sigmoid'))

# Compile the keras model
# Loss Function options: mean_squared_error, binary_crossentropy
# By default, optimizer has a low learning rate, thus, I apply my_adam_opt with a higher rate of learning
# optimaizer='adam' :: Aparentely, has lower oscillations in the loss function, where the learning rate is hi
#my_opt = keras.optimizers.Adam(learning_rate=param_learningrate)
# optimaizer='SGD'  :: Apparently, has higher oscillations in the loss function, where the learning rate is hi
# TODO: what is the impacto of momentum
my_opt = keras.optimizers.SGD(learning_rate=param_learningrate,momentum=0)
model.compile(loss='binary_crossentropy', optimizer=my_opt, metrics=['accuracy'])

if param_test_mode==1:
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # fit the keras model on the dataset
    train_history = model.fit(X, y, epochs=param_epochs, batch_size=10, verbose=1, validation_data=(X, y),callbacks=[tensorboard_callback])
    train_loss = train_history.history['loss']
    train_accuracy = train_history.history['accuracy']
    test_loss = train_history.history['val_loss']
    test_accuracy = train_history.history['val_accuracy']

    rhpplot.plot_train_and_test_loss_accuracy(train_loss,train_accuracy,test_loss,test_accuracy)
else:
    train_loss=[]
    train_accuracy=[]
    test_loss=[]
    test_accuracy=[]
    weight1=[]
    weight2=[]
    for t in range(0,param_epochs):
        tmp = model.fit(X, y, epochs=1, batch_size=10, verbose=1, validation_data=(X, y))
        #Save weights
        for l in model.layers:
            for ws in l.get_weights():
                if len(ws)==2:
                    weight1.append(ws[0])
                    weight2.append(ws[1])
        #save history
        train_loss.append(tmp.history['loss'])
        train_accuracy.append(tmp.history['accuracy'])
        test_loss.append(tmp.history['val_loss'])
        test_accuracy.append(tmp.history['val_accuracy'])

    rhpplot.plot_loss_trajectory(weight1, weight2, train_loss)
    rhpplot.plot_train_and_test_loss_accuracy(train_loss, train_accuracy, test_loss, test_accuracy)

# evaluate the keras model
loss, accuracy = model.evaluate(X, y)
print('\nEvaluate the model::Samples: %d  Loss: %.2f  Accuracy: %.2f' % (len(X), loss * 100, accuracy * 100))

# Prints some tests
print('\nTests')
for a, b in X:
    if a == 1 and b == 1:
        r = 1
    else:
        r = 0
    loss, accuracy = model.evaluate([[a, b]], [r])
    p = model.predict([[a, b]])
    print('%d AND %d == %d  ::: Predicted=%.2f Loss: %.2f  Accuracy: %.2f' % (a, b, r, p, loss * 100, accuracy * 100))


print(model.summary())
