#! /usr/bin/python

from WN_exp import *
from WN_evaluation import *

launch(op='SME_lin', simfn='Dot', ndim=50, nhid=50, marge=1., lremb=0.01,
    lrparam=1., nbatches=100, totepochs=950, test_all=950,
    savepath='WN_SME_lin', datapath='../data/')
# Training takes 4 hours on GTX675M and an intel core i7 processor

print "\n##### EVALUATION #####\n"

ClassifEval(datapath='../data/', loadmodel='WN_SME_lin/best_valid_model.pkl')
RankingEval(datapath='../data/', loadmodel='WN_SME_lin/best_valid_model.pkl')
