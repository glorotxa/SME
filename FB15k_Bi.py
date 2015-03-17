from FB15k_exp import *
from FB15k_evaluation import *

savepath='FB15k_Bi'
datapath='data/'
launch(op='Bi', ndim=100, marge=0.25, lremb=0.01, nbatches=150, dataset='FB',
       totepochs=500, test_all=500, neval=1000, savepath=savepath, datapath=datapath)

print "\n##### EVALUATION #####\n"

MR, T10 = RankingEvalFil(datapath=datapath, dataset='FB', op='Bi',
        loadmodel=savepath+'/best_valid_model.pkl', Nrel=1345, Nsyn=14951)

print "\n##### MEAN RANK: %s #####\n" % (MR)
print "\n##### HITS@10: %s #####\n" % (T10)
