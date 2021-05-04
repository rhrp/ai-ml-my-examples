from random import random

import matplotlib.pyplot as plt
import numpy as np

# Z function - x and y are arrays, so the result is also an array with the same shape
def funParabolic(x, y):
    return x**2+y**2
def funSin(x,y):
    return np.sin(X)
def funTest(x,y):
    return np.sin(X)+np.cos(y)
# Z function - calculated cell by cell
def funA(X,Y):
    out=np.empty([len(X),len(Y)])
    for ix in range(0,len(X)):
        for iy in range(0,len(Y)):
            x=X[ix,iy]
            y=Y[ix,iy]
            z=x**2+y**2
            out[ix,iy]=z
    return out


figure = plt.figure()
axis = figure.add_subplot(111, projection = '3d')

xs = ys = np.arange(-10.0, 10.0, 0.05)
X, Y = np.meshgrid(xs, ys)
Z=funA(X,Y)


axis.plot_surface(X, Y, Z)
axis.set_xlabel('x-axis')
axis.set_ylabel('y-axis')
axis.set_zlabel('z-axis')

plt.show()