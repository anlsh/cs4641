"""Methods for importing the datasets in python-usable format."""

from collections import Counter

import random
import numpy as np
from sklearn import preprocessing
from pybrain.datasets.classification import ClassificationDataSet


def cifar_batches():
    import cPickle
    data = []
    for batch_num in range(1, 6):
        with open('datasets/cifar-10-batches-py/data_batch_' + str(batch_num), 'rb') as f:
            data.append(cPickle.load(f))
    return data

def cifar_test_batch():
    import cPickle
    with open('datasets/cifar-10-batches-py/test_batch', 'rb') as f:
        data = cPickle.load(f)
    return data

def cifar(one_hot=False, ten_percent=False):
    data_ = cifar_batches() + [cifar_test_batch()]
    join_batches = lambda r1, r2, cat: [x for sublist in [batch[cat] for batch in data_[r1:r2]] for x in sublist]
    train_data, train_labels = shuffle_data(join_batches(0, 4, 'data'), join_batches(0, 4, 'labels'), ten_percent=ten_percent)
    test_data, test_labels = shuffle_data(join_batches(4, 6, 'data'), join_batches(4, 6, 'labels'), ten_percent=ten_percent)

    data = {
            'train': {
                'data': preprocessing.normalize(train_data),
                'labels': list(train_labels)
                },
            'test': {
                'data': preprocessing.normalize(test_data),
                'labels': list(test_labels)
                }
            }

    if one_hot:
        for set_ in ['train', 'test']:
            for i in xrange(len(data[set_]['labels'])):
                klass = data[set_]['labels'][i]
                one_hot_vec = [0] * 10
                one_hot_vec[klass] = 1
                data[set_]['labels'][i] = one_hot_vec

    return data

def cifar_nn(offset=None):
    data_ = cifar(one_hot=True, ten_percent=False)
    x_dim = len(data_['train']['data'][0])
    data = ClassificationDataSet(x_dim, 10)
    if offset:
        max_sample = offset
    else:
        max_sample = len(data_['train']['data'])
    for i in xrange(max_sample):
        data.addSample(data_['train']['data'][i], data_['train']['labels'][i])
    data_['train_nn'] = data
    return data_


def shuffle_data(data, labels, ten_percent=False):
    z = zip(data, labels)
    random.shuffle(z)
    if ten_percent:
        z = z[:int(len(z) * 0.1)]
    return zip(*z)


def read_sentiment_data(f_name):
    data = []
    with open('datasets/sentiment_labelled_sentences/' + f_name, 'rb') as f:
        review = ''
        for line in f:
            if line[-2] in ['0', '1']:
                data.append((review + line[:-3], line[-2]))
                review = ''
            else:
                review += line
    return data

def sentiment_imdb():
    return read_sentiment_data('imdb_labelled.txt')

def sentiment_amazon():
    return read_sentiment_data('amazon_cells_labelled.txt')

def sentiment_yelp():
    return read_sentiment_data('yelp_labelled.txt')

def sentiment(bag_size=100):
    data_ = sentiment_imdb() + sentiment_amazon() + sentiment_yelp()

    # Calculate the bag_size most common words.
    all_words = Counter()
    for example, _ in data_:
        for word in example.split(' '):
            all_words[word] += 1
    bag_of_words = all_words.most_common(bag_size)

    # Create features (whether example has each of the words in bag).
    data = []
    labels = []
    word_in_example = lambda word, example: 1 if word in example else 0
    for example, label in data_:
        labels.append(int(label))
        data.append(np.array([word_in_example(word, example) for word, _ in bag_of_words]))

    # Shuffle data and separate into train and test set.
    data, labels = shuffle_data(data, labels)
    offset = int(len(data) * 0.7)
    return {
            'train': {
                'data': data[0:offset],
                'labels': labels[0:offset]
                },
            'test': {
                'data': data[offset:],
                'labels': labels[offset:]
                }
            }

def sentiment_nn(bag_size=100, offset=None):
    data_ = sentiment(bag_size)
    x_dim = len(data_['train']['data'][0])
    data = ClassificationDataSet(x_dim, 1)
    if offset:
        max_sample = offset
    else:
        max_sample = len(data_['train']['data'])
    for i in xrange(max_sample):
        data.addSample(data_['train']['data'][i], [data_['train']['labels'][i]])
    data_['train_nn'] = data
    return data_

