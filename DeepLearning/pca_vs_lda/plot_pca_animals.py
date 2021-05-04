# -*- coding: utf-8 -*-
# This example is very usefull to see the importance of the data quality and the choise of good features
# A dataset of animals described by four features are labeled with 0/1 (live in water) 
# The Whale and the Dolphin have features of animals that dont lives in water, but they live
# On the other side, the Alligator and Crocodile live in the water, but their features say the opposit
# The PCA algoritmn is used to reduce the four dimensions to 2D and 3D
print(__doc__)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import proj3d


param_add_noise=True #add some noise to avoid overlapping 

#load data from a CSV
csv_animals = pd.read_csv('../datasets/animals.csv', header=0, sep=';')

#Select the Data, Target and labels of targets
X = csv_animals.values[:,[1,2,3,4]]
y = csv_animals.values[:,5]
y_labels=csv_animals.values[:,0]
target_names = ['Lives out of the water','Lives in the water']


# Percentage of variance explained for each components
#print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))


#2D ====================================================================
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

#add some noise to avoid overlapping   [n,2]
if param_add_noise:
    l=len(X_r)
    X_r=X_r+np.random.randn(l,2)/50

plt.figure()
colors = ['brown','blue']
linewidth = 2
for i in range(0,len(X)):
    #print('X=%.2f %.2f %.2f %.2f'  % (X[i,0], X[i,1], X[i,2], X[i,3]))
    color=colors[y[i]]
    color=None
    label=y_labels[i]
    #print('i=',i,'  ',y_labels[i],': ',color,'  ',label,X_r[i, 0],'  ', X_r[i, 1])
    plt.scatter(X_r[i, 0], X_r[i, 1], alpha=.8, lw=linewidth,label=label,color=color)
    xannot=X_r[i, 0]
    yannot=X_r[i, 1]
    plt.annotate(
        label,
        xy=(xannot, yannot), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))    

    
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Animals dataset (2D)')
plt.show()


   
#3D ====================================================================
pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)

#add some noise to avoid overlapping [n,1]
if param_add_noise:
    l=len(X_r)
    X_r=X_r+(np.random.randn(l,3)/50)

annots=[]
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=y,cmap=plt.cm.Set1, edgecolor='k', s=40)
for i in range(0,len(X)):
    color=colors[y[i]]
    label=y_labels[i]
    tooltips_xy=(0,0)
    print('i=%d Reduced: %.4f %.4f %.4f   color=%s label=%s'  % (i,X_r[i,0], X_r[i,1], X_r[i,2],color,label))
    if y[i]==1:
        ax.scatter(X_r[i, 0], X_r[i, 1], X_r[i, 2], c=color,cmap=plt.cm.Set1, edgecolor='k', s=20,alpha=1)
        tooltips_xy=(-20,-20)   # this position improves de visibility
    else:
        ax.scatter(X_r[i, 0], X_r[i, 1], X_r[i, 2], c=color,cmap=plt.cm.Set1, edgecolor='k', s=20,alpha=1)
        tooltips_xy=(-20,20)    # this position improves de visibility
    # Annotations
    xannot,yannot,_ = proj3d.proj_transform(X_r[i, 0], X_r[i, 1], X_r[i, 2], ax.get_proj())
    annot=plt.annotate(label,xy=(xannot, yannot), xytext=(tooltips_xy),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')) 
    annots.append(annot)
ax.set_title("First three PCA directions (3D)")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])


#The event for updating annotations
def update_position(e):
    for i in range(0,len(X)):
        xannot,yannot,_ = proj3d.proj_transform(X_r[i, 0], X_r[i, 1], X_r[i, 2], ax.get_proj())
        annots[i].xy = xannot,yannot
        annots[i].update_positions(fig.canvas.renderer)
    fig.canvas.draw()
    print('Update... %d,%d'%(xannot,yannot))
    return True
fig.canvas.mpl_connect('button_release_event', update_position)

plt.show()

