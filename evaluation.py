#! /usr/bin/python

from model import *


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


def RankingEval(datapath='data/', dataset='WordNet3.0-test',
        loadmodel='best_valid_model.pkl', neval='all', Nsyn=40989):

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
    dres.update({'microlp@10': np.mean(np.asarray(res[0]) <= 10) * 10.})
    dres.update({'micrormean': np.mean(res[1])})
    dres.update({'micrormedian': np.median(res[1])})
    dres.update({'microrp@10': np.mean(np.asarray(res[1]) <= 10) * 10.})
    resg = res[0] + res[1]
    dres.update({'microgmean': np.mean(resg)})
    dres.update({'microgmedian': np.median(resg)})
    dres.update({'microgp@10': np.mean(np.asarray(resg) <= 10) * 10.})

    print "### MICRO:"
    print "\t-- left   >> mean: %s, median: %s, p@10: %s%%" % (
            round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
            round(dres['microlp@10'], 3))
    print "\t-- right  >> mean: %s, median: %s, p@10: %s%%" % (
            round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
            round(dres['microrp@10'], 3))
    print "\t-- global >> mean: %s, median: %s, p@10: %s%%" % (
            round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
            round(dres['microgp@10'], 3))

    listrel = set(idxo)
    dictrelres = {}
    dictrellmean = {}
    dictrelrmean = {}
    dictrelgmean = {}
    dictrellmedian = {}
    dictrelrmedian = {}
    dictrelgmedian = {}
    dictrellp10 = {}
    dictrelrp10 = {}
    dictrelgp10 = {}

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
        dictrellp10[i] = np.mean(np.asarray(dictrelres[i][0]) <= 10) * 10.
        dictrelrp10[i] = np.mean(np.asarray(dictrelres[i][1]) <= 10) * 10.
        dictrelgp10[i] = np.mean(np.asarray(dictrelres[i][0] +
                                            dictrelres[i][1]) <= 10) * 10.

    dres.update({'dictrelres': dictrelres})
    dres.update({'dictrellmean': dictrellmean})
    dres.update({'dictrelrmean': dictrelrmean})
    dres.update({'dictrelgmean': dictrelgmean})
    dres.update({'dictrellmedian': dictrellmedian})
    dres.update({'dictrelrmedian': dictrelrmedian})
    dres.update({'dictrelgmedian': dictrelgmedian})
    dres.update({'dictrellp10': dictrellp10})
    dres.update({'dictrelrp10': dictrelrp10})
    dres.update({'dictrelgp10': dictrelgp10})

    dres.update({'macrolmean': np.mean(dictrellmean.values())})
    dres.update({'macrolmedian': np.mean(dictrellmedian.values())})
    dres.update({'macrolp@10': np.mean(dictrellp10.values())})
    dres.update({'macrormean': np.mean(dictrelrmean.values())})
    dres.update({'macrormedian': np.mean(dictrelrmedian.values())})
    dres.update({'macrorp@10': np.mean(dictrelrp10.values())})
    dres.update({'macrogmean': np.mean(dictrelgmean.values())})
    dres.update({'macrogmedian': np.mean(dictrelgmedian.values())})
    dres.update({'macrogp@10': np.mean(dictrelgp10.values())})

    print "### MACRO:"
    print "\t-- left   >> mean: %s, median: %s, p@10: %s%%" % (
            round(dres['macrolmean'], 5), round(dres['macrolmedian'], 5),
            round(dres['macrolp@10'], 3))
    print "\t-- right  >> mean: %s, median: %s, p@10: %s%%" % (
            round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
            round(dres['macrorp@10'], 3))
    print "\t-- global >> mean: %s, median: %s, p@10: %s%%" % (
            round(dres['macrogmean'], 5), round(dres['macrogmedian'], 5),
            round(dres['macrogp@10'], 3))

    idx2lemme = cPickle.load(open('data/idx2lemme.pkl'))
    offset = 0
    if type(embeddings) is list:
        o = o[-embeddings[1].N:, :]
        offset = l.shape[0] - embeddings[1].N
    for i in np.sort(list(listrel)):
        print "### RELATION %s:" % idx2lemme[offset + i]
        print "\t-- left   >> mean: %s, median: %s, p@10: %s%%, N: %s" % (
                round(dictrellmean[i], 5), round(dictrellmedian[i], 5),
                round(dictrellp10[i], 3), len(dictrelres[i][0]))
        print "\t-- right  >> mean: %s, median: %s, p@10: %s%%, N: %s" % (
                round(dictrelrmean[i], 5), round(dictrelrmedian[i], 5),
                round(dictrelrp10[i], 3), len(dictrelres[i][1]))
        print "\t-- global >> mean: %s, median: %s, p@10: %s%%, N: %s" % (
                round(dictrelgmean[i], 5), round(dictrelgmedian[i], 5),
                round(dictrelgp10[i], 3),
                len(dictrelres[i][0] + dictrelres[i][1]))

    return dres

if __name__ == '__main__':
    RankingEval()
