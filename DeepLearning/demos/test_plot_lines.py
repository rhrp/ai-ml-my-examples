from scipy import * #@UnusedWildImport
import matplotlib.pyplot as plt
import numpy as np

plt.gray()
plt.plot(0,0,'.')
plt.show()

x = np.linspace(0, 10, 1000)
plt.plot(x, 2*x, label='m=2')
plt.plot(x, -1*x+4, label='m=-1 b=4')
plt.legend(loc='upper right', frameon=False)
plt.show()
