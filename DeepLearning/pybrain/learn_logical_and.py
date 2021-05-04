import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#
# 1 - One Epoch, 2 - Until Converge, 3/4 - Fixed number of Epochs
#
train_method = 2
n_epochs = 5000
v_learningrate = 0.02
verbose_train = True		# Turns on/Off the verbose during the train


net = buildNetwork(2, 1, 1)
# Print details about the NN
print(net['hidden0'])
print(net)

# Define the dataset with 4 samples
ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (0,))
ds.addSample((1, 0), (0,))
ds.addSample((1, 1), (1,))

print ('Length of the DA:', len(ds))
print ('Data set input')
print (ds['input'])

# Plot the Dataset
plt.gray()
for inp, tgt in ds:
    print(inp[0], ' AND ', inp[1], ' = ', tgt[0])
    if tgt[0] == 1:
        plt.plot(inp[0], inp[1], 'b.')
    else:
        plt.plot(inp[0], inp[1], 'ro')
plt.show()

# Initialize the trainer of the NN based on backpropagation, possible params: verbose=True,momentum=0,learningrate=0.01
# Options: verbose=True,momentum=0, learningrate=0.01
trainer = BackpropTrainer(net, ds, verbose=verbose_train, learningrate=v_learningrate)

if train_method == 1:
	# Train the associated module for one epoch.
	trainer.train()
elif train_method == 2:
	# Creates another DS 4 times bigger in order to allow a proportion 0,75/0,25
	ds_train=ds.copy()
	for i in range(1, 10):
		for inp, tgt in ds:
			ds_train.addSample(inp,tgt)
	print ('Train DS  size=', len(ds_train))
	# Training the Network using the dataset until the convergence hapens
	trnerr, valerr = trainer.trainUntilConvergence(dataset=ds_train, maxEpochs=n_epochs, validationProportion=0.25)
	# Visualizing the error and validation data
	plt.plot(trnerr, 'b', label='Train')
	plt.plot(valerr, 'r', label='Validation')
	plt.title("Train Statistics")
	plt.ylabel("Errors")
	plt.xlabel("Epochs")
	plt.legend(loc='upper right', frameon=False)
	plt.show()
elif train_method==3:
	# will run the loop N times to train it.
	for epoch in range(n_epochs):
		trainer.train()
else:
	# Train on the current dataset for the given number of epochs.
	trainer.trainEpochs(n_epochs)


# Test the NN using the dataset
trainer.testOnData(dataset=ds, verbose=True)

print('\n\nMyTests')
print('false and false = ', net.activate([0, 0]))
print('false and true  = ', net.activate([0, 1]))
print('true  and false = ', net.activate([1, 0]))
print('true  and true  = ', net.activate([1, 1]))



