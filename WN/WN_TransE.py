#! /usr/bin/python
from WN_exp import *
from WN_evaluation import *

launch(op='TransE', dataset='WN', simfn='L1', ndim=20, nhid=20, marge=2., lremb=0.01, lrparam=1.,
    nbatches=100, totepochs=1000, test_all=1, neval=1000, savepath='WN_TransE',
    datapath='../data/', Nent=40961,  Nsyn=40943, Nrel=18)

print "\n##### EVALUATION #####\n"
RankingEval(datapath='../data/', loadmodel='WN_TransE/best_valid_model.pkl')
