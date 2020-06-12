Anish Moorthy's CS 4641 Code Submission

REQUIRED PACKAGES
==================
I use standard packages, so chances are you probably have them all installed
already (they should all come as part of Anaconda 2 as well). However, you can
always ensure that all needed packages are installed by runnning

pip install -r requirements.txt

though you might have to do a

pip install numpy scipy

first if pip complains about package installation order


Code
====================
All of my code is provided as Jupyter notebooks, meaning you can launch them as
you would any other jupyter notebook (google colab is also an option, though see my
note at the very end concerning IMDB). All the code for generating graphs and
stuff is in there too!

All code is meant to be run with Python 3

SKLEARNER NOTEBOOK
==================
The file amoorthy8_sklearner_notebook.ipynb contains all of the code for
learners implemented via sklearn (ie everything but neural networks). You can
just run each cell SEQUENTIALLY (the pip install ones at the top might fail, I
haven't tested them outside of colab but they don't really matter) for the most
part.

The way you select the datasets and learners is to run a specific cell (there is
a seperate cell for each dataset, and one for each type of learner): make sure
you run the right ones corresponding to what you're trying to do. These cells
are pretty clearly labeled.

Each cell corresponding to a learner defines some variables, such as
TEST_OVER_TOTAL, VAL_OVER_TRAIN, NUM_VALIDATIONS, etc. These are pretty
self-explanatory. The most important such variable is hyperparameter_configs,
which is a list of hyperparameter dictionaries which will be evaluated. If you
want to test just a single set of hyperparameters, comment out all but one
dictionary

NEURAL NET NOTEBOOK
===================
The file amoorthy8_neuralnets.ipynb contains the code specific to training
neural networks. Instructions are pretty much the same as above: just make sure
you run the CIFAR/IMDB cell corresponding to what you want to test


IMPORTANT NOTE CONCERNING DATASETS
----------------------------------
Keras will automatically download the CIFAR-10 dataset if it's missing, so you
dont need anything for that besides an internet connection (it only needs to
be downloaded once)

The IMDB dataset can also be automatically downloaded, but the processing I do
on it is kinda slow so doing it from scratch is annoying (it takes about 15 mins)
For this reason, I have provided the "imdb-labels.gz" and "imdb-onehots.gz.npz"
files. If you put these files where python can see them (in the same directory
as wherever you're runnning python from: if you're running it from this directory,
you dont need to touch anything), then my code will auto-load these and skip the
tedious processing step.

If you're running the notebook in colab, then you can upload these two files to
the server by running

from google.colab import files
files.upload()

and you should be good to go :D
