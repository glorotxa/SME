SME
===

1. Overview
-----------------------------------------------------------------

This package proposes scripts using Theano to perform training and evaluation
of the Structured Embeddings model (Bordes et al., AAAI 2011) and of the
Semantic Matching Energy model (Bordes et al., AISTATS 2012) on several
datasets.

Please refers to the following link for more details: 
[Bordes et al., AAAI 2011] 
[Bordes et al., AISTATS 2012]

- model.py : contains the classes and functions to create the different model
             and Theano function (training, evaluation...).
- WordNet3.0_parse.py : preprocess the WordNet data available [here].
- WNexp.py : contains an experiment function to train all the different models
             on the WordNet dataset.
- evaluation.py : contains evaluation functions.
- {experiment_name}_{model_name}.py : runs the best hyperparameters experiment
                                      for a given dataset and a given model.


2. 3rd Party Libraries
-----------------------------------------------------------------

You need to install Theano to use those script. It also requires:
Python >= 2.4, Numpy >=1.5.0, Scipy>=0.8.


3. Installation
-----------------------------------------------------------------

Put the script folder in your PYTHONPATH.


4. Preprocess the data
-----------------------------------------------------------------

Put the absolute path of the wordnet-mlj data (downloaded from [here]) at the
beginning of the WordNet3.0_parse.py script and run it.

5. Run and evaluate a model
-----------------------------------------------------------------

Simply run the corresponding {experiment_name}_{model_name}.py file.
[Note: the training takes several hours].
