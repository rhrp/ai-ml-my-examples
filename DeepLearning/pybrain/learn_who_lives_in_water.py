# Inputs: x1 - Has legs
#         x2 - Can swim
#         x3 - has gills
#         x4 - breathes
# Out:    Lives in the water
from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(4, 1, 1)
#print details about the NN
#print net['hidden0']
#print net

#define the dataset
from pybrain.datasets import SupervisedDataSet
import pandas as pd
#load data from a CSV
csv_original = pd.read_csv('../datasets/animals.csv', header=0, sep=';')
csv = csv_original.copy()

#drop the column name
csv=csv.drop('name',1)
print('\nLen of csv')
print(len(csv))

print('\nCSV')
print(csv);

data=csv.values
print('\nData')
print(data)

#get the N samples of 4 features == array N x 4 
#filter columns  1 to the end 
train_data = data[:,[0,1,2,3]]
#get the targets for each sample == array 1 x N
#filter the first column
train_target = data[:,4]

print('\ntrain_data');
print(train_data)

print('\ntrain_target');
print(train_target)

#create the DS
ds = SupervisedDataSet(4, 1)
for i in range(0, len(train_target)):
	ds.addSample(train_data[i], train_target[i])


#initialize the trainer of the NN based on backpropagation, possible params: verbose=True,momentum=0,learningrate=0.01
from pybrain.supervised.trainers import BackpropTrainer
#verbose=True,momentum=0,learningrate=0.01
trainer = BackpropTrainer(net, ds,verbose=False,learningrate=0.01)

# will run the loop N times to train it.
for epoch in range(1000):
	trainer.train()

#Train on the current dataset for the given number of epochs.
#trainer.trainEpochs(1000)


#Test the NN using the dataset
trainer.testOnData(dataset=ds, verbose = True)


lives_in_water_array=net.activate([1,1,0,1])
lives_in_water=lives_in_water_array[0]
print ('Has legs=',1)
print ('Can swim=',1)
print ('Has gills=',0)
print ('Breathes=',1)
print 'lives in the water=',lives_in_water,'  Answer: ',('Yes' if lives_in_water>0.5 else 'No');

print(csv_original)

#plot 
from scipy import * #@UnusedWildImport
import matplotlib.pyplot as plt

plt.gray()
print('Legs and Swim');
l1=True
l0=True
for inp, tgt in ds:
	print(inp[0],'  ',inp[1],' = ',tgt[0])
	if tgt[0]==1:
        	plt.plot(inp[0],inp[1],'bx',alpha=0.5,label='Lives in the water' if l1 else '')
		l1=False
	else:
	 	plt.plot(inp[0],inp[1],'ro',alpha=0.5,label='Dont lives in the water' if l0 else '')
		l0=False
plt.title("Analyse Legs and Swim")
plt.legend(loc='upper right', frameon=False)
plt.xlabel("Has legs")
plt.ylabel("Can Swim")
plt.show()

print('Swim and Gills')
l1=True
l0=True
for inp, tgt in ds:
        print(inp[1],'  ',inp[2],' = ',tgt[0])
        if tgt[0]==1:
        	plt.plot(inp[1],inp[2],'bx',alpha=0.5,label='Lives in the water' if l1 else '')
        	l1=False
        else:
		plt.plot(inp[1],inp[2],'ro',alpha=0.5,label='Dont lives in the water' if l0 else '')
		l0=False
plt.title("Analyse Swim and Gills")
plt.legend(loc='upper right', frameon=False)
plt.xlabel("Can Swim")
plt.ylabel('Has gills')
plt.show()

#import sklearn
#from sklearn import datasets 
#iris = ds.load_iris() 
#X, y = iris.data, iris.target 

