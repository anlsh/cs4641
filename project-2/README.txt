Anish Moorthy (amoorthy8) CS4641 Project 2: Randomized Optimization
====================================================================

You can download my code through git
{
    git clone https://github.com/pipsqueaker/cs4641-project2.git
}

I provide my code in the form of Jupiter Notebooks. The three problems I choose
are all defined in amoorthy8-p2-three-problems.ipynb: you can reproduce my results
simply by running the cells in sequential order (only run one of the cells under the
"problem selection" phase however). The same process will work for running the neural
network experiments located in amoorthy8-p2-neuralnet.ipynb

Hyperparameters for various algorithms are defined in dictionaries within cells of the
"Algorithm & Hyperparameter Selection" sections of each notebook. You can easily test
just one set of hyperparams by commenting out all other sets in the code (and in fact, 
I've provided the code with all but the optimal hyperparameters commented out already
for each algorithm)

IMPORTANT NOTE CONSIDERING CODE SETUP
--------------------------------------
Please launch jupyter/run python from the directory containing the "imdb-onehots.gz.npz" file. 
If you do not, then my code will automatically download and perform preprocessing on the data, 
which can take around fifteen minutes.

I am using a forked version of the mlrose package. Therefore my code WILL NOT FUNCTION
using the standard package from PyPi which you probably have installed.

To install all the dependencies you need to run my code (including my modified mlrose), please run

{
    pip install -r requirements.txt                                                # Install various required packages
    pip uninstall mlrose                                                           # Uninstall the vanilla version of mlrose
    pip install git+https://github.com/pipsqueaker/mlrose.git@master               # Install my custom mlrose
}

And you will be all set. Note that you will want to uninstall my version of mlrose after
finishing with my project (otherwise other people's code probably wont work)

