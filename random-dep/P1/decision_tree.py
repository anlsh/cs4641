from __future__ import division
from sklearn import tree
from sklearn.metrics import mean_squared_error

import datasets
import numpy as np
import matplotlib.pyplot as plt


##### CIFAR DATASET #####
# Preparing data
cifar_data = datasets.cifar()

### CIFAR - Training trees of different max_depths
max_depth = range(2, 20)
train_err = [0] * len(max_depth)
test_err = [0] * len(max_depth)

for i, d in enumerate(max_depth):
    print 'learning a decision tree with max_depth=' + str(d)
    clf = tree.DecisionTreeClassifier(max_depth=d)
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
plt.title('Decision Trees: Performance x Max Depth')
plt.plot(max_depth, test_err, '-', label='test error')
plt.plot(max_depth, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Mean Square Error')
plt.show()
### ---


### CIFAR - Training trees of different training set sizes (fixed max_depth=8)
train_size = len(cifar_data['train']['data'])
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
MAX_DEPTH = 8
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o)
    clf = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
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
plt.title('Decision Trees: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()
### ---


##### SENTIMENT DATASET #####
### SENTIMENT - Training trees of different bag_sizes (fixed max_depth=8)
MAX_BAG_SIZE = 17499
bag_sizes = range(200, MAX_BAG_SIZE // 2, 200)
train_err = [0] * len(bag_sizes)
test_err = [0] * len(bag_sizes)

for i, b in enumerate(bag_sizes):
    sentiment_data = datasets.sentiment(bag_size=b)

    print 'learning a decision tree with bag_size=' + str(b)
    clf = tree.DecisionTreeClassifier(max_depth=8)
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
plt.title('Decision Trees: Performance x Bag Size')
plt.plot(bag_sizes, test_err, '-', label='test error')
plt.plot(bag_sizes, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Bag Size')
plt.ylabel('Mean Square Error')
plt.show()
### ---


# Preparing data with bag_size=1500
sentiment_data = datasets.sentiment(bag_size=2500)

### SENTIMENT - Training trees of different max_depths
max_depth = range(2, 50)
train_err = [0] * len(max_depth)
test_err = [0] * len(max_depth)

for i, d in enumerate(max_depth):
    print 'learning a decision tree with max_depth=' + str(d)
    clf = tree.DecisionTreeClassifier(max_depth=d)
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
plt.title('Decision Trees: Performance x Max Depth')
plt.plot(max_depth, test_err, '-', label='test error')
plt.plot(max_depth, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Mean Square Error')
plt.show()
### ---


### SENTIMENT - Training trees of different training set sizes (fixed max_depth=35)
train_size = len(sentiment_data['train']['data'])
offsets = range(int(0.1 * train_size), train_size, int(0.03 * train_size))
MAX_DEPTH = 35
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o)
    clf = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
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
plt.title('Decision Trees: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()
### ---
