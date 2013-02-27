import os
import cPickle

import numpy as np
import scipy.sparse as sp

# Put the wordnet-mlj data absolute path here
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
### Creation of the synset/indices dictionnaries

np.random.seed(753)

synlist = []
rellist = []

for datatyp in ['train', 'valid', 'test']:
    f = open(datapath + 'wordnet-mlj12-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        synlist += [lhs[0], rhs[0]]
        rellist += [rel[0]]

synset = np.sort(list(set(synlist)))
relset = np.sort(list(set(rellist)))

synset2idx = {}
idx2synset = {}

idx = 0
for i in synset:
    synset2idx[i] = idx
    idx2synset[idx] = i
    idx += 1
nbsyn = idx
print "Number of synsets in the dictionary: ", nbsyn
# add relations at the end of the dictionary
for i in relset:
    synset2idx[i] = idx
    idx2synset[idx] = i
    idx += 1
nbrel = idx - nbsyn
print "Number of relations in the dictionary: ", nbrel

f = open('../data/WN_synset2idx.pkl', 'w')
g = open('../data/WN_idx2synset.pkl', 'w')
cPickle.dump(synset2idx, f, -1)
cPickle.dump(idx2synset, g, -1)
f.close()
g.close()

####################################################
### Creation of the synset definitions dictionnaries

f = open(datapath + 'wordnet-mlj12-definitions.txt', 'r')
dat = f.readlines()
f.close()

synset2def = {}
synset2concept = {}

for i in dat:
    synset, concept, definition = i[:-1].split('\t')
    synset2def.update({synset: definition})
    synset2concept.update({synset: concept})

f = open('../data/WN_synset2def.pkl', 'w')
g = open('../data/WN_synset2concept.pkl', 'w')
cPickle.dump(synset2def, f, -1)
cPickle.dump(synset2concept, g, -1)
f.close()
g.close()

#################################################
### Creation of the dataset files

for datatyp in ['train', 'valid', 'test']:
    f = open(datapath + 'wordnet-mlj12-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()

    # Declare the dataset variables
    inpl = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(dat)),
            dtype='float32')
    inpr = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(dat)),
            dtype='float32')
    inpo = sp.lil_matrix((np.max(synset2idx.values()) + 1, len(dat)),
            dtype='float32')
    # Fill the sparse matrices
    ct = 0
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        inpl[synset2idx[lhs[0]], ct] = 1
        inpr[synset2idx[rhs[0]], ct] = 1
        inpo[synset2idx[rel[0]], ct] = 1
        ct += 1
    # Save the datasets
    if 'data' not in os.listdir('../'):
        os.mkdir('../data')
    f = open('../data/WN-%s-lhs.pkl' % datatyp, 'w')
    g = open('../data/WN-%s-rhs.pkl' % datatyp, 'w')
    h = open('../data/WN-%s-rel.pkl' % datatyp, 'w')
    cPickle.dump(inpl.tocsr(), f, -1)
    cPickle.dump(inpr.tocsr(), g, -1)
    cPickle.dump(inpo.tocsr(), h, -1)
    f.close()
    g.close()
    h.close()
