# -*- coding: utf-8 -*-

print(__doc__)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import rhplib.datasets_pandas

# load the dataset unsing numpy
#dataset = loadtxt('../datasets/PimaIndiansDiabetesDatabase.csv', skiprows=1, delimiter=',')
## split into input (X) and output (y) variables
#X = dataset[:,0:8]
#y = dataset[:,8]
#features_names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

# load Pima dataset using pandas
dataset = pd.read_csv('../../datasets/PimaIndiansDiabetesDatabase.csv',delimiter=',',header=0)
print(dataset.head())

dataset=rhplib.datasets_pandas.clean_pima_indian_diabetes_dataset(dataset)

# split into input (X) and output (y) variables
X = dataset.values[:,0:8]
y = dataset['Outcome']

features_names = list(dataset.columns.values)
features_names.remove('Outcome')

target_names=['Patient dont has diabetes','Patient has diabetes']

print(features_names)
print(target_names)

pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'% str(pca.explained_variance_ratio_))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
#Plots X as all X_r values where y=0 in column 0, in column 1 for Y and column 2 for Z
#Outcome==0 (when y==0)
ax.scatter(X_r[y == 0,0],X_r[y == 0,1],X_r[y==0,2], c='Green',cmap=plt.cm.Set1, edgecolor='k', s=40,label=target_names[0])
#Outcome==1 (when y==1)
ax.scatter(X_r[y == 1,0],X_r[y == 1,1],X_r[y==1,2], c='Red',cmap=plt.cm.Set1, edgecolor='k', s=40,label=target_names[1])
ax.set_title("PCA of Indian Diabetes dataset")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
ax.legend()
plt.show()


lda = LinearDiscriminantAnalysis(n_components=1)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['Green', 'Red']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 0], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Indian Diabetes dataset')

plt.show()