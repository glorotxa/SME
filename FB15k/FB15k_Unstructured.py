#! /usr/bin/python
from FB15k_exp import *

launch(op='Unstructured', simfn='Dot', ndim=50, nhid=50, marge=1., lremb=0.01,
    lrparam=0.01, nbatches=100, totepochs=1000, test_all=1, neval=1000,
    savepath='FB15k_Unstructured', datapath='../data/')

