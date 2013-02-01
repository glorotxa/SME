#! /usr/bin/python

from WN_exp import *
from WN_evaluation import *

launch(op='SE', simfn='L1', ndim=50, nhid=50, marge=1., lremb=0.01,
    lrparam=1., nbatches=100, totepochs=500, test_all=500, savepath='WN_SE',
    datapath='../data/')
# Training takes 4 hours on GTX675M and an intel core i7 processor

print "\n##### EVALUATION #####\n"

ClassifEval(datapath='../data/', loadmodel='WN_SE/best_valid_model.pkl')
RankingEval(datapath='../data/', loadmodel='WN_SE/best_valid_model.pkl')
