# -*- coding: utf-8 -*-
# This examples shows how LDA can predict a result unsing a supervised dataset
# In this example, we have two classes (live/not live in the water).
# So, the transformation is just possible to one component.
# The dataset is very repetitive (ex: Dog,Cat, Lion and other have the same values on their features).
# Thus, the LDA algoritmn was not capable of faind a good variance. In order to solve this problem, some noise was added to samples.
# The PCA algoritmn is used to reduce the four dimensions to 2D and 3D
print(__doc__)

import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np

#load data from a CSV
csv_animals = pd.read_csv('../../datasets/rhp/animals.csv', header=0, sep=';')

#Select the Data, Target and labels of targets
X = np.asarray(csv_animals.values[:,[1,2,3,4]],dtype=float)
y = np.asarray(csv_animals.values[:,5],dtype=int)
y_labels=csv_animals.values[:,0]
target_names = ['Lives out of the water','Lives in the water']

#add some noise to input data in order to preditct   [n,4]
l=len(X)
X=X+np.random.randn(l,4)/20


#1D ====================================================================
#min(4 , 2-1 )  4 features and 2 classes (live and not live in the water)
lda = LinearDiscriminantAnalysis(n_components=1)
X_reduced = lda.fit(X, y).transform(X)


print('Explained variance ratio: %s' % str(lda.explained_variance_ratio_))

plt.figure()
colors = ['brown','blue']
linewidth = 2
for i in range(0,len(X)):
    color=colors[y[i]]
    #color=None
    label=y_labels[i]
    x1=X_reduced[i, 0]
    x2=0       # Display the dots alined
    #x2=18-i    # Enables a shift from up to down in same order to legend 
    plt.scatter(x1,x2, alpha=.8, lw=linewidth,label=label,color=color)
    #print('i=%d %s\t X=%.2f %.2f %.2f %.2f  :: %.2f  :: Reduced: %.2f'  % (i,label,X[i,0], X[i,1], X[i,2], X[i,3],y[i],X_reduced[i, 0]))
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Animals dataset')
plt.show()

#Has legs;Can swim;Has gills;breathes
p=lda.predict([[0,0,0,1]])
print('p=',p)
