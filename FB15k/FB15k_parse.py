import os, sys
import cPickle

import numpy as np
import scipy.sparse as sp

# Put the freebase15k data absolute path here
datapath = None
assert datapath is not None

if 'data' not in os.listdir('../'):
    os.mkdir('../data')


def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

#################################################
### Creation of the entities/indices dictionnaries

np.random.seed(753)

entleftlist = []
entrightlist = []
rellist = []

for datatyp in ['train']:
    f = open(datapath + 'freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        entleftlist += [lhs[0]]
        entrightlist += [rhs[0]]
        rellist += [rel[0]]

entleftset = np.sort(list(set(entleftlist) - set(entrightlist)))
entsharedset = np.sort(list(set(entleftlist) & set(entrightlist)))
entrightset = np.sort(list(set(entrightlist) - set(entleftlist)))
relset = np.sort(list(set(rellist)))

entity2idx = {}
idx2entity = {}


# we keep the entities specific to one side of the triplets contiguous
idx = 0
for i in entrightset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbright = idx
for i in entsharedset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbshared = idx - nbright
for i in entleftset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbleft = idx - (nbshared + nbright)

print "# of only_left/shared/only_right entities: ", nbleft, '/', nbshared, '/', nbright
# add relations at the end of the dictionary
for i in relset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbrel = idx - (nbright + nbshared + nbleft)
print "Number of relations: ", nbrel

f = open('../data/FB15k_entity2idx.pkl', 'w')
g = open('../data/FB15k_idx2entity.pkl', 'w')
cPickle.dump(entity2idx, f, -1)
cPickle.dump(idx2entity, g, -1)
f.close()
g.close()

#################################################
### Creation of the dataset files

unseen_ents=[]
remove_tst_ex=[]

for datatyp in ['train', 'valid', 'test']:
    print datatyp
    f = open(datapath + 'freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()

    # Declare the dataset variables
    inpl = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)),
            dtype='float32')
    inpr = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)),
            dtype='float32')
    inpo = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)),
            dtype='float32')
    # Fill the sparse matrices
    ct = 0
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        if lhs[0] in entity2idx and rhs[0] in entity2idx and rel[0] in entity2idx: 
            inpl[entity2idx[lhs[0]], ct] = 1
            inpr[entity2idx[rhs[0]], ct] = 1
            inpo[entity2idx[rel[0]], ct] = 1
            ct += 1
        else:
            if lhs[0] in entity2idx:
                unseen_ents+=[lhs[0]]
            if rel[0] in entity2idx:
                unseen_ents+=[rel[0]]
            if rhs[0] in entity2idx:
                unseen_ents+=[rhs[0]]
            remove_tst_ex+=[i[:-1]]

    # Save the datasets
    if 'data' not in os.listdir('../'):
        os.mkdir('../data')
    f = open('../data/FB15k-%s-lhs.pkl' % datatyp, 'w')
    g = open('../data/FB15k-%s-rhs.pkl' % datatyp, 'w')
    h = open('../data/FB15k-%s-rel.pkl' % datatyp, 'w')
    cPickle.dump(inpl.tocsr(), f, -1)
    cPickle.dump(inpr.tocsr(), g, -1)
    cPickle.dump(inpo.tocsr(), h, -1)
    f.close()
    g.close()
    h.close()

unseen_ents=list(set(unseen_ents))
print len(unseen_ents)
remove_tst_ex=list(set(remove_tst_ex))
print len(remove_tst_ex)

for i in remove_tst_ex:
    print i





