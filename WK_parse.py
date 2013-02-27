import os
import cPickle

import numpy as np
import scipy.sparse as sp

# Put the freebase_aaai11 data absolute path here
datapath = '/home/glx/Data/WK/'
assert datapath is not None

if 'data' not in os.listdir('.'):
    os.mkdir('data')


def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

#################################################
### Creation of the entities/indices dictionnaries

np.random.seed(753)

entlist = []
rellist = []

for datatyp in ['train']:
    f = open(datapath + 'wikipedia_mlj12-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        entlist += lhs
        entlist += rhs
        rellist += rel

entset = np.sort(list(set(entlist) - set(rellist)))
sharedset = np.sort(list(set(entlist) & set(rellist)))
relset = np.sort(list(set(rellist) - set(entlist)))

entity2idx = {}
idx2entity = {}


# we keep the entities specific to one side of the triplets contiguous
idx = 0
for i in relset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbrel = idx
for i in sharedset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbshared = idx - nbrel
for i in entset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbent = idx - (nbshared + nbrel)

print "# of ent/shared/rel entities: ", nbent, '/', nbshared, '/', nbrel

f = open('data/WK_entity2idx.pkl', 'w')
g = open('data/WK_idx2entity.pkl', 'w')
cPickle.dump(entity2idx, f, -1)
cPickle.dump(idx2entity, g, -1)
f.close()
g.close()

#################################################
### Creation of the dataset files

for datatyp in ['train', 'valid', 'test']:
    f = open(datapath + 'wikipedia_mlj12-%s.txt' % datatyp, 'r')
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
        for j in lhs:
            inpl[entity2idx[j], ct] = 1 / float(len(lhs))
        for j in rhs:
            inpr[entity2idx[j], ct] = 1 / float(len(rhs))
        for j in rel:
            inpo[entity2idx[j], ct] = 1 / float(len(rel))
        ct += 1
    # Save the datasets
    if 'data' not in os.listdir('.'):
        os.mkdir('data')
    f = open('data/WK-%s-lhs.pkl' % datatyp, 'w')
    g = open('data/WK-%s-rhs.pkl' % datatyp, 'w')
    h = open('data/WK-%s-rel.pkl' % datatyp, 'w')
    cPickle.dump(inpl.tocsr(), f, -1)
    cPickle.dump(inpr.tocsr(), g, -1)
    cPickle.dump(inpo.tocsr(), h, -1)
    f.close()
    g.close()
    h.close()
