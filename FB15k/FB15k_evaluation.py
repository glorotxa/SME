#! /usr/bin/python
import sys
from model import *

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


def RankingEval(datapath='../data/', dataset='FB15k-test',
        loadmodel='best_valid_model.pkl', neval='all', Nsyn=14951, n=10,
        idx2synsetfile='FB15k_idx2entity.pkl'):

    # Load model
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    f.close()

    # Load data
    l = load_file(datapath + dataset + '-lhs.pkl')
    r = load_file(datapath + dataset + '-rhs.pkl')
    o = load_file(datapath + dataset + '-rel.pkl')
    if type(embeddings) is list:
        o = o[-embeddings[1].N:, :]

    # Convert sparse matrix to indexes
    if neval == 'all':
        idxl = convert2idx(l)
        idxr = convert2idx(r)
        idxo = convert2idx(o)
    else:
        idxl = convert2idx(l)[:neval]
        idxr = convert2idx(r)[:neval]
        idxo = convert2idx(o)[:neval]

    ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nsyn)
    rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nsyn)

    res = RankingScoreIdx(ranklfunc, rankrfunc, idxl, idxr, idxo)
    dres = {}
    dres.update({'microlmean': np.mean(res[0])})
    dres.update({'microlmedian': np.median(res[0])})
    dres.update({'microlhits@n': np.mean(np.asarray(res[0]) <= n) * 100})
    dres.update({'micrormean': np.mean(res[1])})
    dres.update({'micrormedian': np.median(res[1])})
    dres.update({'microrhits@n': np.mean(np.asarray(res[1]) <= n) * 100})
    resg = res[0] + res[1]
    dres.update({'microgmean': np.mean(resg)})
    dres.update({'microgmedian': np.median(resg)})
    dres.update({'microghits@n': np.mean(np.asarray(resg) <= n) * 100})

    print "### MICRO:"
    print "\t-- left   >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
            n, round(dres['microlhits@n'], 3))
    print "\t-- right  >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
            n, round(dres['microrhits@n'], 3))
    print "\t-- global >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
            n, round(dres['microghits@n'], 3))

    listrel = set(idxo)
    dictrelres = {}
    dictrellmean = {}
    dictrelrmean = {}
    dictrelgmean = {}
    dictrellmedian = {}
    dictrelrmedian = {}
    dictrelgmedian = {}
    dictrellrn = {}
    dictrelrrn = {}
    dictrelgrn = {}

    for i in listrel:
        dictrelres.update({i: [[], []]})

    for i, j in enumerate(res[0]):
        dictrelres[idxo[i]][0] += [j]

    for i, j in enumerate(res[1]):
        dictrelres[idxo[i]][1] += [j]

    for i in listrel:
        dictrellmean[i] = np.mean(dictrelres[i][0])
        dictrelrmean[i] = np.mean(dictrelres[i][1])
        dictrelgmean[i] = np.mean(dictrelres[i][0] + dictrelres[i][1])
        dictrellmedian[i] = np.median(dictrelres[i][0])
        dictrelrmedian[i] = np.median(dictrelres[i][1])
        dictrelgmedian[i] = np.median(dictrelres[i][0] + dictrelres[i][1])
        dictrellrn[i] = np.mean(np.asarray(dictrelres[i][0]) <= n) * 100
        dictrelrrn[i] = np.mean(np.asarray(dictrelres[i][1]) <= n) * 100
        dictrelgrn[i] = np.mean(np.asarray(dictrelres[i][0] +
                                           dictrelres[i][1]) <= n) * 100

    dres.update({'dictrelres': dictrelres})
    dres.update({'dictrellmean': dictrellmean})
    dres.update({'dictrelrmean': dictrelrmean})
    dres.update({'dictrelgmean': dictrelgmean})
    dres.update({'dictrellmedian': dictrellmedian})
    dres.update({'dictrelrmedian': dictrelrmedian})
    dres.update({'dictrelgmedian': dictrelgmedian})
    dres.update({'dictrellrn': dictrellrn})
    dres.update({'dictrelrrn': dictrelrrn})
    dres.update({'dictrelgrn': dictrelgrn})

    dres.update({'macrolmean': np.mean(dictrellmean.values())})
    dres.update({'macrolmedian': np.mean(dictrellmedian.values())})
    dres.update({'macrolhits@n': np.mean(dictrellrn.values())})
    dres.update({'macrormean': np.mean(dictrelrmean.values())})
    dres.update({'macrormedian': np.mean(dictrelrmedian.values())})
    dres.update({'macrorhits@n': np.mean(dictrelrrn.values())})
    dres.update({'macrogmean': np.mean(dictrelgmean.values())})
    dres.update({'macrogmedian': np.mean(dictrelgmedian.values())})
    dres.update({'macroghits@n': np.mean(dictrelgrn.values())})

    print "### MACRO:"
    print "\t-- left   >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['macrolmean'], 5), round(dres['macrolmedian'], 5),
            n, round(dres['macrolhits@n'], 3))
    print "\t-- right  >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
            n, round(dres['macrorhits@n'], 3))
    print "\t-- global >> mean: %s, median: %s, hits@%s: %s%%" % (
            round(dres['macrogmean'], 5), round(dres['macrogmedian'], 5),
            n, round(dres['macroghits@n'], 3))

    return dres

def RankingEvalFil(datapath='../data/', dataset='umls-test', op='TransE',
        loadmodel='best_valid_model.pkl', fold=0, Nrel=26, Nsyn=135):

    # Load model
    if op == 'TATEC':
        f = open(loadmodel)
        embeddings = cPickle.load(f)
        leftopbi = cPickle.load(f)
        leftoptri = cPickle.load(f)
        rightopbi = cPickle.load(f)
        rightoptri = cPickle.load(f)
        f.close()
    else:
        f = open(loadmodel)
        embeddings = cPickle.load(f)
        leftop =cPickle.load(f)
        rightop =cPickle.load(f)
        f.close()

    # Load data
    l = load_file(datapath + dataset + '-test-lhs.pkl')
    r = load_file(datapath + dataset + '-test-rhs.pkl')
    o = load_file(datapath + dataset + '-test-rel.pkl')
    o = o[-Nrel:, :]

    tl = load_file(datapath + dataset + '-train-lhs.pkl')
    tr = load_file(datapath + dataset + '-train-rhs.pkl')
    to = load_file(datapath + dataset + '-train-rel.pkl')
    to = to[-Nrel:, :]


    vl = load_file(datapath + dataset + '-valid-lhs.pkl')
    vr = load_file(datapath + dataset + '-valid-rhs.pkl')
    vo = load_file(datapath + dataset + '-valid-rel.pkl')
    vo = vo[-Nrel:, :]

    idxl = convert2idx(l)
    idxr = convert2idx(r)
    idxo = convert2idx(o)
    idxtl = convert2idx(tl)
    idxtr = convert2idx(tr)
    idxto = convert2idx(to)
    idxvl = convert2idx(vl)
    idxvr = convert2idx(vr)
    idxvo = convert2idx(vo)
    
    if op == 'Bi':
        ranklfunc = RankLeftFnIdxBi(embeddings, leftop, rightop,
            subtensorspec=Nsyn)
        rankrfunc = RankRightFnIdxBi(embeddings, leftop, rightop,
            subtensorspec=Nsyn)
    elif op == 'Tri':
        ranklfunc = RankLeftFnIdxTri(embeddings, leftop, rightop,
            subtensorspec=Nsyn)
        rankrfunc = RankRightFnIdxTri(embeddings, leftop, rightop,
            subtensorspec=Nsyn)
    elif op == 'TATEC':
        ranklfunc = RankLeftFnIdxTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri,
            subtensorspec=Nsyn)
        rankrfunc = RankRightFnIdxTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri,
            subtensorspec=Nsyn)        
    
    true_triples=np.concatenate([idxtl,idxvl,idxl,idxto,idxvo,idxo,idxtr,idxvr,idxr]).reshape(3,idxtl.shape[0]+idxvl.shape[0]+idxl.shape[0]).T

    restest = FilteredRankingScoreIdx(ranklfunc, rankrfunc, idxl, idxr, idxo, true_triples)
    T10 = np.mean(np.asarray(restest[0] + restest[1]) <= 10) * 100
    MR = np.mean(np.asarray(restest[0] + restest[1]))

    return MR, T10

if __name__ == '__main__':
    RankingEval(loadmodel=sys.argv[1])
