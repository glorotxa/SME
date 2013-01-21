SME
===

1. Overview
-----------------------------------------------------------------

This package proposes scripts using Theano to perform training and evaluation
of the Structured Embeddings model (Bordes et al., AAAI 2011) and of the
Semantic Matching Energy model (Bordes et al., AISTATS 2012) on several
datasets.

Please refer to the following paper for more details: 
https://www.hds.utc.fr/everest/lib/exe/fetch.php?id=en%3Asmemlj12&cache=cache&media=en:bordes12aistats.pdf

- model.py : contains the classes and functions to create the different model
             and Theano function (training, evaluation...).
- WordNet3.0_parse.py : parses the WordNet data available at:
                        https://www.hds.utc.fr/everest/doku.php?id=en:smemlj12.
- WNexp.py : contains an experiment function to train all the different models
             on the WordNet dataset.
- evaluation.py : contains evaluation functions.
- in the WN/ folder:
	* {experiment_name}_{model_name}.py : runs the best hyperparameters
					      experiment for a given dataset
                                              and a given model.
	* {experiment_name}_{model_name}.out : output of the best
					       hyperparameters experiment for a
					       given dataset and a given model.
	* {experiment_name}_test.py : perform quick runs for small models of
	                              all types to test the scripts.


2. 3rd Party Libraries
-----------------------------------------------------------------

You need to install Theano to use those scripts. It also requires:
Python >= 2.4, Numpy >=1.5.0, Scipy>=0.8.

The experiment scripts are compatible with Jobman but this library is not
mandatory.


3. Installation
-----------------------------------------------------------------

Put the script folder in your PYTHONPATH.


4. Create the data files
-----------------------------------------------------------------

Put the absolute path of the extracted wordnet-mlj data (downloaded from:
https://www.hds.utc.fr/everest/doku.php?id=en:smemlj12) at the beginning of the
WordNet3.0_parse.py script and run it (the SME folder has to be your current
directory).

5. Run and evaluate a model
-----------------------------------------------------------------

Simply run the corresponding {experiment_name}_{model_name}.py file (the
SME/WN/ folder has to be your current directory).  [Note: the training takes
several hours].
