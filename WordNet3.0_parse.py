import os
import cPickle

import numpy as np
import scipy.sparse as sp

# Put the wordnet-mlj data absolute path here
datapath = None
assert datapath is not None

if 'data' not in os.listdir('.'):
    os.mkdir('data')

#################################################
### Creation of the lemma to synset dictionnaries
f = open(datapath + 'WordNet3.0-filtered-lemma2synsets+cnts.txt', 'r')
dat = f.readlines()
f.close()

lemme2synset = {}
lemme2freq = {}

for idx, i in enumerate(dat):
    lemme, synsets, frequence = i[:-1].split('\t')
    if synsets[0] != '_':
        synsets = synsets[:-1]
        frequence = frequence[:-1]
    synlist = synsets.split(' ')
    freqlist = list(np.asarray(frequence.split(' '), dtype='float64'))
    lemme2synset.update({lemme: synlist})
    lemme2freq.update({lemme: freqlist})

## Add ConceptNet relations
relations = ['_PropertyOf', '_MadeOf', '_DefinedAs', '_PartOf', '_IsA',
             '_UsedFor', '_CapableOfReceivingAction', '_LocationOf',
             '_SubeventOf', '_LastSubeventOf', '_PrerequisiteEventOf',
             '_FirstSubeventOf', '_EffectOf', '_DesirousEffectOf',
             '_DesireOf', '_MotivationOf', '_CapableOf']
for i in relations:
    lemme2synset.update({i: [i]})
    lemme2freq.update({i: [1.0]})

# Save
f = open('data/lemme2synset.pkl', 'w')
g = open('data/lemme2freq.pkl', 'w')
cPickle.dump(lemme2synset, f, -1)
cPickle.dump(lemme2freq, g, -1)
f.close()
g.close()

#################################################
### Creation of the synset to lemma dictionnaries
f = open(datapath + 'WordNet3.0-filtered-synset2lemmas.txt', 'r')
dat = f.readlines()
f.close()

synset2lemme = {}

for idx, i in enumerate(dat):
    synset, lemmes = i[:-1].split('\t')
    lemmes = lemmes[:-1]
    lemmelist = lemmes.split(' ')
    synset2lemme.update({synset: lemmelist})

# Add relations
for j in lemme2synset.keys():
    if j[1] != '_':
        synset2lemme.update({j: [j]})

f = open('data/synset2lemme.pkl', 'w')
cPickle.dump(synset2lemme, f, -1)
f.close()

#################################################
### Creation of the indexes correspondances
synset2idx = {}
idx2synset = {}
lemme2idx = {}
idx2lemme = {}

idx = 0

# Synsets
for i in synset2lemme.keys():
    if i[0] != '_':
        synset2idx.update({i: idx})
        idx2synset.update({idx: i})
        idx += 1

# Lemmes
for i in lemme2synset.keys():
    if i[1] == '_':
        # if not a relation
        if len(lemme2synset[i]) > 1:
            # if it has more than one synset
            lemme2idx.update({i: idx})
            idx2lemme.update({idx: i})
            idx += 1
        else:
            # if not we merge both
            lemme2idx.update({i: synset2idx[lemme2synset[i][0]]})

## Relations
for i in lemme2synset.keys():
    if i[1] != '_':
        lemme2idx.update({i: idx})
        idx2lemme.update({idx: i})
        idx += 1

f = open('data/lemme2idx.pkl', 'w')
g = open('data/idx2lemme.pkl', 'w')
h = open('data/synset2idx.pkl', 'w')
l = open('data/idx2synset.pkl', 'w')
cPickle.dump(lemme2idx, f, -1)
cPickle.dump(idx2lemme, g, -1)
cPickle.dump(synset2idx, h, -1)
cPickle.dump(idx2synset, l, -1)
f.close()
g.close()
h.close()
l.close()


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


f = open('data/synset2def.pkl', 'w')
g = open('data/synset2concept.pkl', 'w')
cPickle.dump(synset2def, f, -1)
cPickle.dump(synset2concept, g, -1)
f.close()
g.close()


####################################################
### Creation of the negative synsets correspondances
f = open(datapath + 'WordNet3.0-filtered-synset2negative_synsets.txt', 'r')
dat = f.readlines()
f.close()

synset2neg = {}

for idx, i in enumerate(dat):
    synset, neg = i[:-1].split('\t')
    neg = neg[:-1]
    synset2neg.update({synset: neg.split(' ')})

f = open('data/synset2neg.pkl', 'w')
cPickle.dump(synset2neg, f, -1)
f.close()


##################################################
### Creation of the concept synset correspondances
f = open(datapath + 'WordNet3.0-filtered-oldname2synset.txt', 'r')
dat = f.readlines()
f.close()

concept2synset = {}

for i in dat:
    concept, synset = i[:-1].split('\t')
    concept2synset.update({concept: synset})

f = open('data/concept2synset.pkl', 'w')
cPickle.dump(concept2synset, f, -1)
f.close()


##################################################
### Dataset creation
def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

np.random.seed(753)

for options in ['normal', 'bridge', 'ambiguated']:
    if options == 'normal':
        typelist = ['train', 'valid', 'test']
    else:
        typelist = ['train']
    for datatyp in typelist:
        f = open(datapath + 'wordnet-mlj12-%s.txt' % datatyp, 'r')
        dat = f.readlines()
        f.close()
        # Count the number of examples
        ct = 0
        for i in dat:
            lhs, rel, rhs = parseline(i[:-1])
            if options == 'normal' or options == 'ambiguated':
                ct += 1
            if options == 'bridge':
                for j in synset2lemme[lhs[0]]:
                    if len(lemme2synset[j]) != 1:
                        ct += 1
                for j in synset2lemme[rhs[0]]:
                    if len(lemme2synset[j]) != 1:
                        ct += 1
        print options, datatyp, np.max(lemme2idx.values()) + 1, ct
        # Declare the dataset variables
        inpl = sp.lil_matrix((np.max(lemme2idx.values()) + 1, ct),
                dtype='float32')
        inpr = sp.lil_matrix((np.max(lemme2idx.values()) + 1, ct),
                dtype='float32')
        inpo = sp.lil_matrix((np.max(lemme2idx.values()) + 1, ct),
                dtype='float32')
        # Fill the sparse matrices
        ct = 0
        for i in dat:
            lhs, rel, rhs = parseline(i[:-1])
            if options == 'normal':
                inpl[synset2idx[lhs[0]], ct] = 1
                inpr[synset2idx[rhs[0]], ct] = 1
                inpo[lemme2idx[rel[0]], ct] = 1
                ct += 1
            if options == 'bridge':
                for j in synset2lemme[lhs[0]]:
                    if len(lemme2synset[j]) != 1:
                        inpl[lemme2idx[j], ct] = 1
                        inpr[synset2idx[rhs[0]], ct] = 1
                        inpo[lemme2idx[rel[0]], ct] = 1
                        ct += 1
                for j in synset2lemme[rhs[0]]:
                    if len(lemme2synset[j]) != 1:
                        inpr[lemme2idx[j], ct] = 1
                        inpl[synset2idx[lhs[0]], ct] = 1
                        inpo[lemme2idx[rel[0]], ct] = 1
                        ct += 1
            if options == 'ambiguated':
                tmplist = synset2lemme[lhs[0]]
                tmpidx = lemme2idx[tmplist[np.random.randint(len(tmplist))]]
                inpl[tmpidx, ct] = 1
                tmplist = synset2lemme[rhs[0]]
                tmpidx = lemme2idx[tmplist[np.random.randint(len(tmplist))]]
                inpr[tmpidx, ct] = 1
                inpo[lemme2idx[rel[0]], ct] = 1
                ct += 1
        # Save the datasets
        if 'data' not in os.listdir('.'):
            os.mkdir('data')
        if options == 'normal':
            f = open('data/WordNet3.0-%s-lhs.pkl' % datatyp, 'w')
            g = open('data/WordNet3.0-%s-rhs.pkl' % datatyp, 'w')
            h = open('data/WordNet3.0-%s-rel.pkl' % datatyp, 'w')
        else:
            f = open('data/WordNet3.0-%s-%s-lhs.pkl' % (options, datatyp), 'w')
            g = open('data/WordNet3.0-%s-%s-rhs.pkl' % (options, datatyp), 'w')
            h = open('data/WordNet3.0-%s-%s-rel.pkl' % (options, datatyp), 'w')
        cPickle.dump(inpl.tocsr(), f, -1)
        cPickle.dump(inpr.tocsr(), g, -1)
        cPickle.dump(inpo.tocsr(), h, -1)
        f.close()
        g.close()
        h.close()
