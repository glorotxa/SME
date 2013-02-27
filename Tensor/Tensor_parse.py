import os
import cPickle

import numpy
import scipy.sparse

# Number of folds
K = 10
datapath = None
assert datapath is not None

if 'data' not in os.listdir('../'):
        os.mkdir('../data')

for dataset in ['kinships', 'umls', 'nations']:
    f = open(datapath + dataset + '.pkl')
    dictdata = cPickle.load(f)
    tensordata = dictdata['tensor']

    # List non-zeros
    lnz = []
    # List zeros
    lz = []
    # List of feature triplets
    if dataset == 'nations':
        lzfeat = []
        lnzfeat = []
    # Fill the lists
    for i in range(tensordata.shape[0]):
        for j in range(tensordata.shape[1]):
            for k in range(tensordata.shape[2]):
                # Separates features triplets for nation
                if dataset == 'nations' and (i >= 14 or j >= 14):
                    if tensordata[i, j, k] == 0:
                        lzfeat += [(i, j, k)]
                    elif tensordata[i, j, k] == 1:
                        lnzfeat += [(i, j, k)]
                else:
                    if tensordata[i, j, k] == 0:
                        lz += [(i, j, k)]
                    elif tensordata[i, j, k] == 1:
                        lnz += [(i, j, k)]

    # Pad the feature triplets lists (same for all training folds)
    if dataset == 'nation':
        if len(lzfeat) < len(lnzfeat):
            while len(lzfeat) < len(lnzfeat):
                lzfeat += lzfeat[:len(lnzfeat) - len(lzfeat)]
        else:
            while len(lnzfeat) < len(lzfeat):
                lnzfeat += lnzfeat[:len(lzfeat) - len(lnzfeat)]

    f = open(datapath + dataset + '_permutations.pkl')
    idxnz = cPickle.load(f)
    idxz = cPickle.load(f)
    f.close()

    # For each fold
    for k in range(K):
        if k != K - 1:
            tmpidxnz = (idxnz[:k * len(idxnz) / K] +
                        idxnz[(k + 2) * len(idxnz) / K:])
            tmpidxz = (idxz[:k * len(idxz) / K] +
                       idxz[(k + 2) * len(idxz) / K:])
            tmpidxtestnz = idxnz[k * len(idxnz) / K:(k + 1) * len(idxnz) / K]
            tmpidxtestz = idxz[k * len(idxz) / K:(k + 1) * len(idxz) / K]
            tmpidxvalnz = idxnz[(k + 1) * len(idxnz) / K:
                                (k + 2) * len(idxnz) / K]
            tmpidxvalz = idxz[(k + 1) * len(idxz) / K:(k + 2) * len(idxz) / K]
        else:
            tmpidxnz = idxnz[len(idxnz) / K:k * len(idxnz) / K]
            tmpidxz = idxz[len(idxz) / K:k * len(idxz) / K]
            tmpidxtestnz = idxnz[k * len(idxnz) / K:(k + 1) * len(idxnz) / K]
            tmpidxtestz = idxz[k * len(idxz) / K:(k + 1) * len(idxz) / K]
            tmpidxvalnz = idxnz[:len(idxnz) / K]
            tmpidxvalz = idxz[:len(idxz) / K]

        # Test data files
        testl = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], len(tmpidxtestnz) + len(tmpidxtestz)))
        testr = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], len(tmpidxtestnz) + len(tmpidxtestz)))
        testo = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], len(tmpidxtestnz) + len(tmpidxtestz)))
        outtest = []
        ct = 0
        for j in tmpidxtestnz:
            i = lnz[j]
            testl[i[0], ct] = 1
            testr[i[1], ct] = 1
            testo[i[2] + tensordata.shape[1], ct] = 1
            outtest += [1]
            ct += 1
        for j in tmpidxtestz:
            i = lz[j]
            testl[i[0], ct] = 1
            testr[i[1], ct] = 1
            testo[i[2] + tensordata.shape[1], ct] = 1
            outtest += [0]
            ct += 1
        f = open('../data/%s-test-lhs-fold%s.pkl' % (dataset, k), 'w')
        g = open('../data/%s-test-rhs-fold%s.pkl' % (dataset, k), 'w')
        h = open('../data/%s-test-rel-fold%s.pkl' % (dataset, k), 'w')
        l = open('../data/%s-test-targets-fold%s.pkl' % (dataset, k), 'w')
        cPickle.dump(testl.tocsr(), f, -1)
        cPickle.dump(testr.tocsr(), g, -1)
        cPickle.dump(testo.tocsr(), h, -1)
        cPickle.dump(numpy.asarray(outtest), l, -1)
        f.close()
        g.close()
        h.close()
        l.close()

        # Valid data files
        validl = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], len(tmpidxvalnz) + len(tmpidxvalz)))
        validr = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], len(tmpidxvalnz) + len(tmpidxvalz)))
        valido = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], len(tmpidxvalnz) + len(tmpidxvalz)))
        outvalid = []
        ct = 0
        for j in tmpidxvalnz:
            i = lnz[j]
            validl[i[0], ct] = 1
            validr[i[1], ct] = 1
            valido[i[2] + tensordata.shape[1], ct] = 1
            outvalid += [1]
            ct += 1
        for j in tmpidxvalz:
            i = lz[j]
            validl[i[0], ct] = 1
            validr[i[1], ct] = 1
            valido[i[2] + tensordata.shape[1], ct] = 1
            outvalid += [0]
            ct += 1
        f = open('../data/%s-valid-lhs-fold%s.pkl' % (dataset, k), 'w')
        g = open('../data/%s-valid-rhs-fold%s.pkl' % (dataset, k), 'w')
        h = open('../data/%s-valid-rel-fold%s.pkl' % (dataset, k), 'w')
        l = open('../data/%s-valid-targets-fold%s.pkl' % (dataset, k), 'w')
        cPickle.dump(validl.tocsr(), f, -1)
        cPickle.dump(validr.tocsr(), g, -1)
        cPickle.dump(valido.tocsr(), h, -1)
        cPickle.dump(numpy.asarray(outvalid), l, -1)
        f.close()
        g.close()
        h.close()
        l.close()

        # Train data files
        # Pad the shorter list
        if len(tmpidxz) < len(tmpidxnz):
            while len(tmpidxz) < len(tmpidxnz):
                tmpidxz += tmpidxz[:len(tmpidxnz) - len(tmpidxz)]
        else:
            while len(tmpidxnz) < len(tmpidxz):
                tmpidxnz += tmpidxnz[:len(tmpidxz) - len(tmpidxnz)]

        ct = len(tmpidxz)
        if dataset == 'nations':
            ct += len(lzfeat)
        trainposl = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], ct))
        trainnegl = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], ct))
        trainposr = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], ct))
        trainnegr = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], ct))
        trainposo = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], ct))
        trainnego = scipy.sparse.lil_matrix((tensordata.shape[1] +
            tensordata.shape[2], ct))
        ct = 0
        for u, v in zip(tmpidxnz, tmpidxz):
            ipos = lnz[u]
            ineg = lz[v]
            trainposl[ipos[0], ct] = 1
            trainnegl[ineg[0], ct] = 1
            trainposr[ipos[1], ct] = 1
            trainnegr[ineg[1], ct] = 1
            trainposo[ipos[2] + tensordata.shape[1], ct] = 1
            trainnego[ineg[2] + tensordata.shape[1], ct] = 1
            ct += 1
        # Add all the feature triplets to each folds
        if dataset == 'nations':
            for u, v in zip(lnzfeat, lzfeat):
                ipos = u
                ineg = v
                trainposl[ipos[0], ct] = 1
                trainnegl[ineg[0], ct] = 1
                trainposr[ipos[1], ct] = 1
                trainnegr[ineg[1], ct] = 1
                trainposo[ipos[2] + tensordata.shape[1], ct] = 1
                trainnego[ineg[2] + tensordata.shape[1], ct] = 1
                ct += 1
        f = open('../data/%s-train-pos-lhs-fold%s.pkl' % (dataset, k), 'w')
        g = open('../data/%s-train-pos-rhs-fold%s.pkl' % (dataset, k), 'w')
        h = open('../data/%s-train-pos-rel-fold%s.pkl' % (dataset, k), 'w')
        l = open('../data/%s-train-neg-lhs-fold%s.pkl' % (dataset, k), 'w')
        m = open('../data/%s-train-neg-rhs-fold%s.pkl' % (dataset, k), 'w')
        n = open('../data/%s-train-neg-rel-fold%s.pkl' % (dataset, k), 'w')
        cPickle.dump(trainposl.tocsr(), f, -1)
        cPickle.dump(trainposr.tocsr(), g, -1)
        cPickle.dump(trainposo.tocsr(), h, -1)
        cPickle.dump(trainnegl.tocsr(), l, -1)
        cPickle.dump(trainnegr.tocsr(), m, -1)
        cPickle.dump(trainnego.tocsr(), n, -1)
        f.close()
        g.close()
        h.close()
        l.close()
        m.close()
        n.close()
