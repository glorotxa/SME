#! /usr/bin/python

from Tensor_exp import *
from Tensor_evaluation import *

model_list = ['SE','Unstructured','SME_bil','SME_lin']
data_list = ['umls', 'kinships', 'nations']
fold_list = [0]#,1,2,3,4,5,6,7,8,9]

for m in model_list:
    for d in data_list:
        for f in fold_list:
            print "\n----- %s ----- %s ----- %s\n"  % (m,d,f)
            if m == 'SE':
                simfn = 'L1'
            else:
                simfn = 'Dot'
            if d == 'umls':
                Nrel = 49
                Nent = 184
                nbatches = 60
            if d == 'kinships':
                Nrel = 26
                Nent = 130
                nbatches = 60
            if d == 'nations':
                Nrel = 57
                Nent = 182
                nbatches = 4
            launch(op=m, simfn=simfn, ndim=2, nhid=3, marge=1., lremb=0.01,
                    lrparam=1., nbatches=nbatches, totepochs=2, test_all=1, 
                    savepath='test', datapath='../data/', dataset=d,
                    fold=f, Nrel=Nrel, Nent=Nent)

            print "\n##### EVALUATION %s ----- %s ----- %s #####\n" % (m,d,f)
            PRAUCEval(datapath='../data/', dataset=d + '-test', fold=f,
                    loadmodel='test/best_valid_model.pkl')

