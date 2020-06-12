Raphael Gontijo Lopes
903062081

CS 4641 - Machine Learning

A description of the datasets used can be found in the analysis pdf
The links for their source is below:

CIFAR-10 (python version)
https://www.cs.toronto.edu/~kriz/cifar.html

Sentiment Labelled Sentences
https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

The datasets are included in my submission, along with my code. If you think
it's necessary to redownload them, please put them under, respectively:
datasets/cifar-10-batches-py/
datasets/sentiment_labelled_sentences/

All of the parsing is done in the file datasets/__init__.py

To run any of the experiments mentioned in the analysis, you must have
installed python 2.7, along with numpy, scikit-learn, matplotlib, and pybrain.
Then you can simply:

$ python neural_networks.py

Note that many of the training sessions can take a long time. If you want to run
just one of the experiments, you can modify the source by commenting out in
between the markers:

### SENTIMENT - Training on different set sizes
# code
# you
# can
# comment
# out
### ---

Alternatively, all of my experiment runs are being submitted under experiment/.
The naming scheme is as follows:
nn_cifar_setsize.txt -> stdout dump from running neural networks on CIFAR.
nn_cifar_setsize.png -> a graph showing the results of the experiment.
