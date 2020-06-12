Raphael Gontijo Lopes
903062081

CS 4641 - Machine Learning

A description of the datasets used can be found in the analysis pdf
The links for their source is below:

Sentiment Labelled Sentences
https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

Mushrooms Dataset
https://archive.ics.uci.edu/ml/datasets/Mushroom

The datasets are included in my submission, along with my code.
They have been minimally preprocessed for ease of parsing.
The sentiment dataset has already been through word-bagging.

To run any of the experiments mentioned in the analysis, you must have
installed python 2.7, along with numpy, scikit-learn, matplotlib, and pybrain.
Then you can simply:

$ python analysis.py sentiment
or
$ python analysis.py mushrooms

Note that many of the training sessions can take a long time. If you want to run
just one of the experiments, you can modify the source by commenting out the
function calls at the end of the file.

Alternatively, all of my experiment runs are being submitted under
sentiment_plots/ and mushroom_plots/.
The naming scheme is as follows:
NN.png -> Learning curve for Neural Net
NNwKBtr -> NeuralNet with KBest preprocessing
etc.
