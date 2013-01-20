#! /usr/bin/python

from WNexp import *
from evaluation import *

launch(op='SME_bil', simfn='Dot', ndim=50, nhid=50, marge=1., lremb=0.01,
    lrparam=1., nbatches=100, totepochs=500, test_all=500, savepath='WN_SME_bil')
# Training takes 4 hours on GTX675M and an intel core i7 processor

print "\n##### EVALUATION #####\n"

ClassifEval(loadmodel='WN_SME_bil/best_valid_model.pkl')
RankingEval(loadmodel='WN_SME_bil/best_valid_model.pkl')
