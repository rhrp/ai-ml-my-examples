import matplotlib.pyplot as plt
#
# These functions returns a tuple of X,y
# X - List of features
# y - List of labels for each feature

def makeLogicalAND(gates):
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]
    return X,y

def makeLogicalOR(gates):
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 1]
    return X,y

def makeLogicalXOR(gates):
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    return X,y

# Plot the Dataset
def plotBinaryDataSet(ds_X,ds_y):
    figure=plt.figure()
    plt.gray()
    #print('Plot Dataset')
    for n in range(0,len(ds_X)):
        x=ds_X[n]
        #print(x[0], ' AND ', x[1], ' = ', y[n])
        if ds_y[n] == 1:
            plt.plot(x[0], x[1], 'b.')
        else:
            plt.plot(x[0], x[1], 'ro')
    plt.title('Dataset')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



