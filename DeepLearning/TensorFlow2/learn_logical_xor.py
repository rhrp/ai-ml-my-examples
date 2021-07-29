#
# Logical AND function
#
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from keras.layers import LeakyReLU
from tensorflow import keras
import datetime
import rhplib.plot_functions as rhpplot
import rhplib.datasets_bin as rhpds

param_epochs=1000
param_learningrate=0.01
param_use_leakrelu=True

# split into input (X) and output (y) variables
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

rhpds.plotBinaryDataSet(X,y)

# define the keras model
model = Sequential()
model.add(InputLayer(2))
if param_use_leakrelu:
    model.add(Dense(2))
    model.add(LeakyReLU(alpha=0.3))
else:
    model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the keras model
my_opt = keras.optimizers.SGD(learning_rate=param_learningrate,momentum=0)
model.compile(loss='binary_crossentropy', optimizer=my_opt, metrics=['accuracy'])


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# fit the keras model on the dataset
train_history = model.fit(X, y, epochs=param_epochs, batch_size=10, verbose=1, validation_data=(X, y),callbacks=[tensorboard_callback])
train_loss = train_history.history['loss']
train_accuracy = train_history.history['accuracy']
test_loss = train_history.history['val_loss']
test_accuracy = train_history.history['val_accuracy']
rhpplot.plot_train_and_test_loss_accuracy(train_loss,train_accuracy,test_loss,test_accuracy)

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
    print('%d XOR %d == %d  ::: Predicted=%.2f Loss: %.2f  Accuracy: %.2f' % (a, b, r, p, loss * 100, accuracy * 100))


print(model.summary())
