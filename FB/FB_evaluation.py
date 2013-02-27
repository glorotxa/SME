#! /usr/bin/python

from model import *


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


def RankingEval(datapath='../data/', dataset='FB-test',
        loadmodel='best_valid_model.pkl', neval='all', Nright=8309,
        Nshared=7785, n=100, idx2synsetfile='FB_idx2entity.pkl'):

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

    rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nright + Nshared)

    res = RankingScoreRightIdx(rankrfunc, idxl, idxr, idxo)
    dres = {}
    dres.update({'micrormean': np.mean(res)})
    dres.update({'micrormedian': np.median(res)})
    dres.update({'microrr@n': np.mean(np.asarray(res) <= n) * 100})

    print "### MICRO:"
    print "\t-- right  >> mean: %s, median: %s, r@%s: %s%%" % (
            round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
            n, round(dres['microrr@n'], 3))

    listrel = set(idxo)
    dictrelres = {}
    dictrelrmean = {}
    dictrelrmedian = {}
    dictrelrrn = {}

    for i in listrel:
        dictrelres.update({i: []})

    for i, j in enumerate(res):
        dictrelres[idxo[i]] += [j]


    for i in listrel:
        dictrelrmean[i] = np.mean(dictrelres[i])
        dictrelrmedian[i] = np.median(dictrelres[i])
        dictrelrrn[i] = np.mean(np.asarray(dictrelres[i]) <= n) * 100

    dres.update({'dictrelres': dictrelres})
    dres.update({'dictrelrmean': dictrelrmean})
    dres.update({'dictrelrmedian': dictrelrmedian})
    dres.update({'dictrelrrn': dictrelrrn})

    dres.update({'macrormean': np.mean(dictrelrmean.values())})
    dres.update({'macrormedian': np.mean(dictrelrmedian.values())})
    dres.update({'macrorr@n': np.mean(dictrelrrn.values())})

    print "### MACRO:"
    print "\t-- right  >> mean: %s, median: %s, r@%s: %s%%" % (
            round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
            n, round(dres['macrorr@n'], 3))

    idx2synset = cPickle.load(open(datapath + idx2synsetfile))
    offset = 0
    if type(embeddings) is list:
        o = o[-embeddings[1].N:, :]
        offset = l.shape[0] - embeddings[1].N
    for i in np.sort(list(listrel)):
        print "### RELATION %s:" % idx2synset[offset + i]
        print "\t-- right  >> mean: %s, median: %s, r@%s: %s%%, N: %s" % (
                round(dictrelrmean[i], 5), round(dictrelrmedian[i], 5),
                n, round(dictrelrrn[i], 3), len(dictrelres[i]))
    return dres


def ClassifEval(datapath='../data/', validset='FB-valid', testset='FB-test',
        loadmodel='best_valid_model.pkl', seed=647):

    # Load model
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    f.close()

    np.random.seed(seed)

    # Load data
    lv = load_file(datapath + validset + '-lhs.pkl')
    lvn = lv
    rv = load_file(datapath + validset + '-rhs.pkl')
    rvn = rv[:, np.random.permutation(lv.shape[1])]
    ov = load_file(datapath + validset + '-rel.pkl')
    ovn = ov
    if type(embeddings) is list:
        ov = ov[-embeddings[1].N:, :]
        ovn = ovn[-embeddings[1].N:, :]

    # Load data
    lt = load_file(datapath + testset + '-lhs.pkl')
    ltn = lt
    rt = load_file(datapath + testset + '-rhs.pkl')
    rtn = rt[:, np.random.permutation(lv.shape[1])]
    ot = load_file(datapath + testset + '-rel.pkl')
    otn = ot
    if type(embeddings) is list:
        ot = ot[-embeddings[1].N:, :]
        otn = otn[-embeddings[1].N:, :]

    simfunc = SimFn(simfn, embeddings, leftop, rightop)

    resv = simfunc(lv, rv, ov)[0]
    resvn = simfunc(lvn, rvn, ovn)[0]
    rest = simfunc(lt, rt, ot)[0]
    restn = simfunc(ltn, rtn, otn)[0]

    # Threshold
    perf = 0
    T = 0
    for val in list(np.concatenate([resv, resvn])):
        tmpperf = (resv > val).sum() + (resvn <= val).sum()
        if tmpperf > perf:
            perf = tmpperf
            T = val
    testperf = ((rest > T).sum() + (restn <= T).sum()) / float(2 * len(rest))
    print "### Classification performance : %s%%" % round(testperf * 100, 3)

    return testperf


if __name__ == '__main__':
    ClassifEval()
    RankingEval()
