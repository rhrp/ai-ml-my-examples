import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(12, 10))
#ax = Axes3D(fig, elev=-150, azim=110) # Works, but dont show the title!
#111 means 1 tall, 1 wide, plot number 1
ax = fig.add_subplot(111, projection='3d')
ax.scatter(0,0,0, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.scatter(0,4,0, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.scatter(2,0,2, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Plot 3D dots")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

#ax.annotate('0.25 on data', (0, 0), textcoords='data', size=20)

ax.text(0,0,0,  '0,0,0', size=20, zorder=1,   color='k') 
ax.text(0,4,0,  '0,4,0', size=20, zorder=1,   color='k') 
ax.text(2,0,2,  '2,0,2', size=20, zorder=1,   color='k') 


plt.show()
