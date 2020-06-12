from __future__ import division
from sklearn.metrics import mean_squared_error
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

import datasets
import numpy as np
import matplotlib.pyplot as plt


##### CIFAR DATASET #####
### CIFAR - Training NNs for different training set sizes
cifar_data = datasets.cifar_nn()

train_size = len(cifar_data['train']['data'])
offsets = range(int(0.1 * train_size), int(train_size), int(0.1 * train_size))

inp_len = len(cifar_data['test']['data'][0])
out_len = 10
NET_SHAPE = (inp_len, inp_len//2, inp_len//4, out_len)
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning a neural net with training_set_size=' + str(o)
    print 'getting data',
    cifar_data = datasets.cifar_nn(offset=o)
    print 'building net',
    net = buildNetwork(*NET_SHAPE)
    print 'training',
    trainer = BackpropTrainer(net, cifar_data['train_nn'])
    trainer.trainOnDataset(cifar_data['train_nn'], 5)
    print 'validating'
    train_err[i] = mean_squared_error(cifar_data['train']['labels'],
                                      [net.activate(cifar_data['train']['data'][i])
                                          for k in xrange(len(cifar_data['train']['data']))])
    test_err[i] = mean_squared_error(cifar_data['test']['labels'],
                                     [net.activate(cifar_data['test']['data'][i])
                                         for k in xrange(len(cifar_data['test']['data']))])
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('Neural Nets: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()
### ---


##### SENTIMENT DATASET #####
### SENTIMENT - Training NNs of different bag_sizes (fixed n_hidden = 1)
MAX_BAG_SIZE = 17499
bag_sizes = range(200, MAX_BAG_SIZE // 7, 200)
train_err = [0] * len(bag_sizes)
test_err = [0] * len(bag_sizes)

for i, b in enumerate(bag_sizes):
    print 'learning a neural net with bag_size=' + str(b)
    print 'getting data',
    sentiment_data = datasets.sentiment_nn(bag_size=b)
    print 'building net',
    inp_len = len(sentiment_data['test']['data'][0])
    out_len = 1
    NET_SHAPE = (inp_len, inp_len//2, out_len)
    net = buildNetwork(*NET_SHAPE)
    print 'training',
    trainer = BackpropTrainer(net, sentiment_data['train_nn'])
    trainer.trainOnDataset(sentiment_data['train_nn'], 10)
    print 'validating'
    train_err[i] = mean_squared_error(sentiment_data['train']['labels'],
                                      [net.activate(sentiment_data['train']['data'][i])
                                          for k in xrange(len(sentiment_data['train']['data']))])
    test_err[i] = mean_squared_error(sentiment_data['test']['labels'],
                                     [net.activate(sentiment_data['test']['data'][i])
                                         for k in xrange(len(sentiment_data['test']['data']))])
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('Neural Nets: Performance x Bag Size')
plt.plot(bag_sizes, test_err, '-', label='test error')
plt.plot(bag_sizes, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Bag Size')
plt.ylabel('Mean Square Error')
plt.show()
### ---


# Preparing data with bag_size=1000
sentiment_data = datasets.sentiment_nn(bag_size=1000)

### SENTIMENT - Training nns of different n_hidden
inp_len = len(sentiment_data['test']['data'][0])
out_len = 1
net_shapes = [(inp_len, out_len), (inp_len, inp_len//2, out_len),
              (inp_len, inp_len//2, inp_len//4, out_len),
              (inp_len, inp_len//2, inp_len//4, inp_len//5, out_len),
              (inp_len, inp_len//2, inp_len//4, inp_len//5, inp_len//6, out_len)]
train_err = [0] * len(net_shapes)
test_err = [0] * len(net_shapes)

for i, d in enumerate(net_shapes):
    print 'learning a Neural Net with net_shape=' + str(d)
    print 'building net',
    net = buildNetwork(*d)
    print 'training',
    trainer = BackpropTrainer(net, sentiment_data['train_nn'])
    trainer.trainOnDataset(sentiment_data['train_nn'], 10)
    print 'validating'
    train_err[i] = mean_squared_error(sentiment_data['train']['labels'],
                                      [net.activate(sentiment_data['train']['data'][i])
                                          for k in xrange(len(sentiment_data['train']['data']))])
    test_err[i] = mean_squared_error(sentiment_data['test']['labels'],
                                     [net.activate(sentiment_data['test']['data'][i])
                                         for k in xrange(len(sentiment_data['test']['data']))])
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('Neural Nets: Performance x Num Hidden')
plt.plot(map(lambda x: len(x) - 2, net_shapes), test_err, '-', label='test error')
plt.plot(map(lambda x: len(x) - 2, net_shapes), train_err, '-', label='train error')
plt.legend()
plt.xlabel('Num Hidden')
plt.ylabel('Mean Square Error')
plt.show()
### ---


### SENTIMENT - Training NNs of different training set sizes (fixed n_hidden=2)
sentiment_data = datasets.sentiment(bag_size=1000)

train_size = len(sentiment_data['train']['data'])
offsets = range(int(0.1 * train_size), train_size, int(0.03 * train_size))

inp_len = len(sentiment_data['test']['data'][0])
out_len = 1
NET_SHAPE = (inp_len, inp_len//2, inp_len//4, out_len)
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'

for i, o in enumerate(offsets):
    print 'learning a neural net with training_set_size=' + str(o)
    print 'getting data',
    sentiment_data = datasets.sentiment_nn(bag_size=1000, offset=o)
    print 'building net',
    net = buildNetwork(*NET_SHAPE)
    print 'training',
    trainer = BackpropTrainer(net, sentiment_data['train_nn'])
    trainer.trainOnDataset(sentiment_data['train_nn'], 10)
    print 'validating'
    train_err[i] = mean_squared_error(sentiment_data['train']['labels'],
                                      [net.activate(sentiment_data['train']['data'][i])
                                          for k in xrange(len(sentiment_data['train']['data']))])
    test_err[i] = mean_squared_error(sentiment_data['test']['labels'],
                                     [net.activate(sentiment_data['test']['data'][i])
                                         for k in xrange(len(sentiment_data['test']['data']))])
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('Neural Nets: Performance x Training Set Size')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
plt.show()
### ---
