# k-means clustering
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


param_n_clusters=64
param_show_digit=None   # None, 0..9

ds_mnist = pd.read_csv('/home/rhp/MyDevTools/tmp_datasets/mnist_train.csv')

# get the labels
labels = ds_mnist['label']

# Drop the label feature in order to get only the data in the dataset
data = ds_mnist.drop("label",axis=1)

# Data structure
print(data.head())
print('Data: ',data.shape)
print('Labels: ',labels.shape)


# plot the first 4 images of one of the digits
n=0
if param_show_digit in range(0,10):
    for i in range(0,len(labels)):
        if labels[i]==param_show_digit and n<3:
            plt.figure(figsize=(7,7))
            grid_data = data.iloc[i].to_numpy().reshape(28,28)  # reshape from 1d to 2d pixel array
            plt.imshow(grid_data, interpolation = "none", cmap = "gray")
            plt.show()
            n=n+1


# define the model
#model = KMeans(n_clusters=10)
model = MiniBatchKMeans(n_clusters=param_n_clusters)
# fit the model
model.fit(data)

print('Sum of squared distances of samples to their closest cluster center :: Inertia =',model.inertia_)
print('Model\'s labels: ',model.labels_)

# assign a cluster to each example - in each cell the respective cluster of the samples
y_clusters = model.predict(data)
print('y_clusters: ',y_clusters.shape,'     ',y_clusters)

# retrieve unique clusters - the unique clusters 0..9
clusters = np.unique(y_clusters)
print('clusters: ',clusters.shape,'     ',clusters)


# record centroid values
centroids = model.cluster_centers_
print('centroids: ',centroids.shape,'     len=',len(centroids))

from collections import Counter
def most_frequent_label(ListLabels):
    occurence_count = Counter(ListLabels)
    return occurence_count.most_common(1)[0][0]



#https://medium.datadriveninvestor.com/k-means-clustering-for-imagery-analysis-56c9976f16b6
#
predicted_labels=np.zeros(len(labels))
#
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix_tuple = np.where(y_clusters == cluster)
    row_ix=row_ix_tuple[0]
    # get the most frequent digit for this cluster
    most_freq_label=most_frequent_label(labels[row_ix])
    # update predited_digits for this cluster
    #TODO: dont work np.where(yhat == cluster,predited_digits,most_freq_label)
    for i in row_ix:
        predicted_labels[i]=most_freq_label
    #print('cluster=',cluster,'row_ix=',row_ix)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(centroids[cluster].reshape(28,28), interpolation = "none", cmap = "gray")
    ax1.set_title('Centroid')
    sns.histplot(labels[row_ix],bins=10,stat='count',ax=ax2)
    ax2.set_title('Cluster '+str(cluster)+'  most frequent digit '+str(most_freq_label))
    plt.show()
    

#Here the confusion matrix dont work, because we need to associate a label to each cluster
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(predicted_labels, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,xticklabels='x',yticklabels='y')
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()


# Calculate the Accuracy
correct_predicts=0
total_samples=data.shape[0]
for i in range(0,total_samples):
    if (predicted_labels[i]==labels[i]):
        correct_predicts=correct_predicts+1
accuracy=correct_predicts/total_samples*100
print('Total of samples: %d\nCorrect predictions: %d\nAccuracy: %.3f' % (total_samples,correct_predicts,accuracy))