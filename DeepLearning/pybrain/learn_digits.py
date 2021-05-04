import matplotlib.pyplot as plt
from numpy import ravel
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from sklearn import datasets

# 1797 images 8x8 pixels of digits 0 to 9
digits = datasets.load_digits()
X, y = digits.data, digits.target

print(digits.data.shape)
plt.gray()
#show the first ten images
#for i in range(0,9):
#    print i
#    plt.matshow(digits.images[i])
#    plt.show()

ds = ClassificationDataSet(64, 1, nb_classes=10)
print('Len of X and Y ', len(X), len(y))
# Adds the 1797 samples (image 8x8) and its numerical value (0-9)
for i in range(len(X)):
    print (i, '  image 8x8 = ', y[i])
    ds.addSample(ravel(X[i]), y[i])

# Produce data for testing and traing in a proportion of 25%
test_data_temp, training_data_temp = ds.splitWithProportion(0.25)

# Creates a DS for testing
test_data = ClassificationDataSet(64, 1, nb_classes=10)
for n in range(0, test_data_temp.getLength()):
    test_data.addSample(test_data_temp.getSample(n)[0],
                        test_data_temp.getSample(n)[1])

# Creates a DS for training
training_data = ClassificationDataSet(64, 1, nb_classes=10)
for n in range(0, training_data_temp.getLength()):
    training_data.addSample(training_data_temp.getSample(n)[0],
                            training_data_temp.getSample(n)[1])

# ???
test_data._convertToOneOfMany()
training_data._convertToOneOfMany()

# Input Layer: training_data.indim
# Hidden Layer: 64
# Output Layer: training_data.outdim
print('BuildNetwork: Input Layer=', training_data.indim, 'Hidden=', 64, '  Output=', training_data.outdim)
net = buildNetwork(training_data.indim, 64, training_data.outdim, outclass=SoftmaxLayer)

# Traing the NN
trainer = BackpropTrainer(net, dataset=training_data, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)

trnerr, valerr = trainer.trainUntilConvergence(dataset=training_data, maxEpochs=10)
plt.plot(trnerr, 'b', valerr, 'r')
plt.show()

trainer.trainEpochs(10)

# Test the NN
print('Percent Error on testData:', percentError(trainer.testOnClassData(dataset=test_data), test_data['class']))
