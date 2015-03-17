from FB15k_exp import *
from FB15k_evaluation import *


##lremb: learning rate for the parameters of the bigrams model
##lrparam: learning rate for the parameters of the trigram model
launch(op='TATEC', marge=0.25, rhoL=5, lremb=0.001, lrparam=0.001, nbatches=150, dataset='FB',
       totepochs=500, test_all=50, neval=1000, savepath='FB15k_TATEC', datapath='data/', loadmodel=True,
       loadmodelBi='FB15k_Bi/best_valid_model.pkl', loadmodelTri='FB15k_Tri/best_valid_model.pkl')


print "\n##### EVALUATION #####\n"

MR, T10 = RankingEvalFil(datapath='data/', dataset='FB', op='TATEC',
        loadmodel='FB15k_TATEC/best_valid_model.pkl', Nrel=1345, Nsyn=14951)

print "\n##### MEAN RANK: %s #####\n" % (MR)
print "\n##### HITS@10: %s #####\n" % (T10)
