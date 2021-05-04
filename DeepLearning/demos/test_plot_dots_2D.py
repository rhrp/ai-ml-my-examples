import matplotlib.pyplot as plt
import numpy as np

plt.gray()
plt.plot(0,0,'.')
plt.plot(1,1,'.')
plt.plot(2,2,'.')

plt.annotate('1,1',xy=(1, 1), xytext=(-20, 2),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))    

plt.show()

