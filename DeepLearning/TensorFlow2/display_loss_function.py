#
# Display the 3D surface representing the loss function of a Perceptron with two features w1 and w2, as well as a bias
#
#
#
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from tensorflow import keras

from keras import constraints
from rhplib import activation_functions as rhpaf
from rhplib import loss_functions as rhplf
from rhplib import datasets_bin as rhpds

# When using bias, the path of optimization is not aligned with the surface of Loss,
# because the Bias is not fixed in the fit() method as happens in the function funLoss()
# Otherwise, when not using bias, the value is Zero, but de optimization is not the best
param_use_bias=True
param_show_path=True
param_dataset="AND"     # AND, OR or XOR
param_epochs=100

# This function returns a tuple with two values:  loss surface and the optimal loss point (another tuple)
# x - Array of inputs  2*N
# y - Expected output  1*N
# w1 and w2 - The weights
# bias - float
# function
# function
def funLossSurface(X,Y,W1,W2,bias,loss_function,activation_function):
    dim_matrix= W1.shape[0] # W1 and W2 are identical in terms of shape, which is a square NxN
    loss_surface=np.empty([dim_matrix,dim_matrix],float)
    n=len(X)
    optimal_loss_w1_w2_bias_accuracy=(99999,0,0,bias,0)
    for i in range(0,dim_matrix):
        for j in range(0,dim_matrix):
            w1=float(W1[i,j])
            w2=float(W2[i,j])
            P=[]
            correct_predictions=0
            for k in range(0,n):
                x=X[k]
                y=Y[k]
                #print(x,'    ',y)
                x1=float(x[0])
                x2=float(x[1])
                p=w1*x1+w2*x2+bias
                p=activation_function(p)
                if y==1 and p>=0.5 or y==0 and p<0.5:
                    correct_predictions=correct_predictions+1
                print(x1,'  '+param_dataset+' ',x2,' =  ',y,'     p=%2f' %(p))
                P.append(p)
            l=loss_function(Y,P)
            accuracy=100*correct_predictions/n
            print('loss=%.2f  accuracy=%.2f'%(l,accuracy))
            if l<optimal_loss_w1_w2_bias_accuracy[0]:
                optimal_loss_w1_w2_bias_accuracy=(l,w1,w2,bias,accuracy)
            loss_surface[i,j]=l
        print('Optimal Loss=%.2f  w1=%.3f   w2=%.3f  bias=%.3f Accuracy=%d' % optimal_loss_w1_w2_bias_accuracy)
    return loss_surface,optimal_loss_w1_w2_bias_accuracy



# DataSet
if param_dataset == "AND":
    ds_X, ds_y = rhpds.makeLogicalAND(2)
    ds_name='Logical AND function'
elif param_dataset == "OR":
    ds_X, ds_y = rhpds.makeLogicalOR(2)
    ds_name = 'Logical OR function'
elif param_dataset == "XOR":
    ds_X, ds_y = rhpds.makeLogicalXOR(2)
    ds_name = 'Logical XOR function'
else:
    print('Invalid DataSet')
    exit(0)

# define the keras model
model = Sequential()
model.add(InputLayer(2))
model.add(Dense(1, activation='sigmoid',use_bias=param_use_bias))

my_opt = keras.optimizers.SGD(learning_rate=5,momentum=0)
model.compile(loss='binary_crossentropy', optimizer=my_opt, metrics=['accuracy'])

# Evaluate the model: saving the weights, bias, loss and accuracy
history=[]
bias=0
for t in range(0, param_epochs):
    tmp = model.fit(ds_X, ds_y, epochs=1, batch_size=10, verbose=1, validation_data=(ds_X, ds_y))
    # Save loss and accuracy
    loss=tmp.history['loss'][0]
    accuracy=tmp.history['accuracy'][0]
    # Save weights
    w1=0
    w2=0
    for layr in model.layers:
        for ws in layr.get_weights():
            if len(ws) == 2:
                w1=ws[0]
                w2=ws[1]
            if len(ws) == 1:
                bias=ws[0]
                print('Use the bias=',bias)
    # Add to history
    h=(w1,w2,bias,loss,accuracy)
    history.append(h)

# Calculate the surface of Loss
graph_xs=[]
graph_ys=[]
graph_zs=[]
precision=20
delta=30
for i in range(0,precision):
    xv=delta*i/precision-delta/2
    yv=delta*i/precision-delta/2
    graph_xs.append(xv)
    graph_ys.append(yv)
graph_W1, graph_W2 = np.meshgrid(graph_xs, graph_ys)
Loss,optimal=funLossSurface(ds_X,ds_y,graph_W1,graph_W2,bias,rhplf.BinaryCrossentropy,rhpaf.sigmoid)

# Show the dataset
rhpds.plotBinaryDataSet(ds_X,ds_y)

# Show the Loss surface
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(graph_W1,graph_W2, Loss, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.75)

# Customize the z axis.
ax.set_zlim(0, 3.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.title("Loss function of a Neuronal Network while learning a "+ds_name)
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')

# Plot the optimal point (W1, W2 and Loss)
ax.plot(optimal[1], optimal[2], optimal[0],'r+',label='Optimal Loss=%.3f    W1=%.2f   W2=%.2f  Bias=%.2f Accuracy=%d'%optimal)

# Plot history
if param_show_path:
    n=0
    for h in history:
        n=n+1
        if n==len(history): # Last item, the most
            label='Optimization path at the end of %d Epochs (loss=%.3f accuracy=%d)'%(n,h[3],h[4]*100)
        else:
            label=''
        ax.plot(h[0], h[1], h[3],'g.', alpha=0.75, label=label if label != '' else '')

ax.legend(loc='lower right', frameon=True)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
