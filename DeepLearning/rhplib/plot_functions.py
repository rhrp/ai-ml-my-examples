import matplotlib.pyplot as plt
import numpy as np

#
#
#
#
def plot_loss_trajectory(weight1,weight2,train_loss):
    # Represents the path of optimization
    figure = plt.figure()
    axis = figure.add_subplot(111, projection = '3d')

    ws1=np.array(weight1).reshape(len(weight1))
    ws2=np.array(weight2).reshape(len(weight2))
    ls=np.array(train_loss).reshape(len(train_loss))

    axis.plot(ws1,ws2,ls, label='Loss curve')
    axis.scatter(weight1,weight2,train_loss,marker='+')
    #axis.scatter(ws1,ws2,ls, marker='+')

    axis.set_xlabel('Weight 1')
    axis.set_ylabel('Weight 2')
    axis.set_zlabel('Loss')
    axis.margins(x=0, y=-0)
    axis.legend()

    plt.show()
    return

#
#
#
#
def plot_train_and_test_loss_accuracy(train_loss,train_accuracy,test_loss,test_accuracy):
    figure = plt.figure()
    plt.plot(train_loss)
    plt.plot(train_accuracy)
    plt.plot(test_loss)
    plt.plot(test_accuracy)
    plt.legend(['Train loss', 'Train accuracy', 'Test loss', 'Test accuracy'])
    plt.show()
    return

def plot_train_loss_accuracy(train_loss,train_accuracy):
    figure = plt.figure()
    plt.plot(train_loss)
    plt.plot(train_accuracy)
    plt.legend(['Train loss', 'Train accuracy'])
    plt.show()
    return
