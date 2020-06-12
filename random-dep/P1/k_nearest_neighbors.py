from __future__ import division
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

import datasets
import numpy as np
import matplotlib.pyplot as plt


##### CIFAR DATASET #####
# Preparing data
cifar_data = datasets.cifar()

### CIFAR - Training kNNs of different Ks
ks = range(2, 7)
train_err = [0] * len(ks)
test_err = [0] * len(ks)

for i, k in enumerate(ks):
    print 'learning a kNN classifier with k=' + str(k)
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf = clf.fit(cifar_data['train']['data'], cifar_data['train']['labels'])

    train_err[i] = mean_squared_error(cifar_data['train']['labels'],
                                     clf.predict(cifar_data['train']['data']))
    test_err[i] = mean_squared_error(cifar_data['test']['labels'],
                                    clf.predict(cifar_data['test']['data']))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('KNNClassifier: Performance x K')
plt.plot(ks, test_err, '-', label='test error')
plt.plot(ks, train_err, '-', label='train error')
plt.legend()
plt.xlabel('K')
plt.ylabel('Mean Square Error')
plt.show()
### ---


### CIFAR - Training kNNs of different training set sizes (fixed n_neighbors=5)
train_size = len(cifar_data['train']['data'])
offsets = range(int(0.1 * train_size), train_size, int(0.1 * train_size))
N_NEIGHBORS = 5
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning a kNN classifier with training_set_size=' + str(o)
    clf = neighbors.KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    clf = clf.fit(cifar_data['train']['data'][:o], cifar_data['train']['labels'][:o])

    train_err[i] = mean_squared_error(cifar_data['train']['labels'][:o],
            clf.predict(cifar_data['train']['data'][:o]))
    test_err[i] = mean_squared_error(cifar_data['test']['labels'][:o],
                                     clf.predict(cifar_data['test']['data'][:o]))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('KNN CLassifier: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()
### ---


##### SENTIMENT DATASET #####
### SENTIMENT - Training kNNs of different bag_sizes (fixed n_neighbors=5)
MAX_BAG_SIZE = 17499
bag_sizes = range(200, MAX_BAG_SIZE // 2, 200)
train_err = [0] * len(bag_sizes)
test_err = [0] * len(bag_sizes)

for i, b in enumerate(bag_sizes):
    sentiment_data = datasets.sentiment(bag_size=b)

    print 'learning a kNN classifier with bag_size=' + str(b) + ' (fixed n_neighbors=5)'
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    clf = clf.fit(sentiment_data['train']['data'], sentiment_data['train']['labels'])

    train_err[i] = mean_squared_error(sentiment_data['train']['labels'],
                                     clf.predict(sentiment_data['train']['data']))
    test_err[i] = mean_squared_error(sentiment_data['test']['labels'],
                                    clf.predict(sentiment_data['test']['data']))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('KNN Classifier: Performance x Bag Size')
plt.plot(bag_sizes, test_err, '-', label='test error')
plt.plot(bag_sizes, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Bag Size')
plt.ylabel('Mean Square Error')
plt.show()
### ---


# Preparing data with bag_size=1500
sentiment_data = datasets.sentiment(bag_size=2000)

### SENTIMENT - Training kNN of different Ks
ks = range(3, 8)
train_err = [0] * len(ks)
test_err = [0] * len(ks)

for i, k in enumerate(ks):
    print 'learning a kNN classifier with n_neighbors=' + str(k)
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf = clf.fit(sentiment_data['train']['data'], sentiment_data['train']['labels'])

    train_err[i] = mean_squared_error(sentiment_data['train']['labels'],
                                     clf.predict(sentiment_data['train']['data']))
    test_err[i] = mean_squared_error(sentiment_data['test']['labels'],
                                    clf.predict(sentiment_data['test']['data']))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('KNN Classifier: Performance x K')
plt.plot(ks, test_err, '-', label='test error')
plt.plot(ks, train_err, '-', label='train error')
plt.legend()
plt.xlabel('K')
plt.ylabel('Mean Square Error')
plt.show()
### ---


### SENTIMENT - Training kNN of different training set sizes (fixed n_neighbors=5)
train_size = len(sentiment_data['train']['data'])
offsets = range(int(0.1 * train_size), train_size, int(0.03 * train_size))
N_NEIGHBORS = 5
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning a kNN classifier with training_set_size=' + str(o) + ' (fixed n_neighbors=5)'
    clf = neighbors.KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    clf = clf.fit(sentiment_data['train']['data'][:o], sentiment_data['train']['labels'][:o])

    train_err[i] = mean_squared_error(sentiment_data['train']['labels'][:o],
            clf.predict(sentiment_data['train']['data'][:o]))
    test_err[i] = mean_squared_error(sentiment_data['test']['labels'][:o],
                                     clf.predict(sentiment_data['test']['data'][:o]))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('KNN CLassifier: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()
### ---
