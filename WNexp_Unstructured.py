#! /usr/bin/python

from WNexp import *
from evaluation import *

launch(op='Unstructured', simfn='Dot', ndim=50, nhid=50, marge=1., lremb=0.1,
    lrparam=0.1, nbatches=100, totepochs=720, test_all=720,
    savepath='WN_Unstructured')
# Training takes 2 hours on GTX675M and an intel core i7 processor

print "\n##### EVALUATION #####\n"

ClassifEval(loadmodel='WN_Unstructured/best_valid_model.pkl')
RankingEval(loadmodel='WN_Unstructured/best_valid_model.pkl')
