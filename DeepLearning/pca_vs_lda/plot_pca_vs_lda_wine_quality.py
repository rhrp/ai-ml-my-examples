# -*- coding: utf-8 -*-

print(__doc__)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy
from mpl_toolkits.mplot3d import Axes3D

# load Winequality dataset using pandas
dataset = pd.read_csv('../../datasets/winequality-red.csv',delimiter=',',header=0)
print(dataset.head())

#fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality
# split into input (X) and output (y) variables
X = dataset.values[:,0:11]
y = dataset.values[:,11]
features_names = list(dataset.columns.values)
features_names.remove('quality')

print(features_names)


pca = PCA(n_components=3)
X_pca = pca.fit(X).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'% str(pca.explained_variance_ratio_))

print('len(X_r)=',len(X_pca))



fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
for qs in numpy.unique(y):
    q=int(qs)
    label='Quality '+str(q)
    ax.scatter(X_pca[y == q,0],X_pca[y == q,1],X_pca[y==q,2] ,cmap=plt.cm.Set1, edgecolor='k', s=40,label=label)
    print('q=',q,'  total values ',numpy.count_nonzero(y == q))
ax.set_title("PCA of Wine Quality dataset")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
ax.legend()
plt.show()


lda = LinearDiscriminantAnalysis(n_components=3)
X_lda = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
for qs in numpy.unique(y):
    q=int(qs)
    label='Quality '+str(q)
    ax.scatter(X_lda[y == q,0],X_lda[y == q,1],X_lda[y==q,2] ,cmap=plt.cm.Set1, edgecolor='k', s=40,label=label)
    print('q=',q,'  total values ',numpy.count_nonzero(y == q))
ax.set_title("LDA of Wine Quality dataset")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
ax.legend()
plt.show()