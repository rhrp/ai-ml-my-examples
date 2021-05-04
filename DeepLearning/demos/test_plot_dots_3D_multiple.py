import matplotlib.pyplot as plt

fig = plt.figure(1, figsize=(12, 10))
#ax = Axes3D(fig, elev=-150, azim=110) # Works, but dont show the title!
#111 means 1 tall, 1 wide, plot number 1
ax = fig.add_subplot(331, projection='3d')
ax.scatter(0,0,0, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.scatter(0,1,0, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.scatter(3,1,6, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Plot 3D dots (plot 1)")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

from mpl_toolkits.mplot3d import proj3d
x2, y2, _ = proj3d.proj_transform(3,1,6, ax.get_proj())
annot=plt.annotate('3,1,6',xy=(x2, y2), xytext=(-20, 2),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))   

bx = fig.add_subplot(335, projection='3d')
bx.scatter(0,0,0, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
bx.scatter(0,1,0, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
bx.set_title("Plot 3D dots (plot 5)")
bx.set_xlabel("1st eigenvector")
bx.w_xaxis.set_ticklabels([])
bx.set_ylabel("2nd eigenvector")
bx.w_yaxis.set_ticklabels([])
bx.set_zlabel("3rd eigenvector")
bx.w_zaxis.set_ticklabels([])


cx = fig.add_subplot(339, projection='3d')
cx.scatter(0,0,0, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
cx.scatter(3,1,6, c=None,cmap=plt.cm.Set1, edgecolor='k', s=40)
cx.set_title("Plot 3D dots (plot 9)")
cx.set_xlabel("1st eigenvector")
cx.w_xaxis.set_ticklabels([])
cx.set_ylabel("2nd eigenvector")
cx.w_yaxis.set_ticklabels([])
cx.set_zlabel("3rd eigenvector")
cx.w_zaxis.set_ticklabels([])




#The event for dot 3,1,6 at ax plot
def update_position(e):
    x2, y2, _ = proj3d.proj_transform(3,1,6, ax.get_proj())
    annot.xy = x2,y2
    annot.update_positions(fig.canvas.renderer)
    fig.canvas.draw()
    print('Update... %d,%d'%(x2,y2))
    return True
fig.canvas.mpl_connect('button_release_event', update_position)


plt.show()