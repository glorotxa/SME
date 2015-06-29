import cPickle
import gzip
import os

import numpy
import theano
import scipy.sparse


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[numpy.argsort(cols)]

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)

def create_random_arr(shape, Nent):
    """
    This function create a random sparse index matrix with a given shape. It
    is useful to create negative triplets.

    :param shape: shape of the desired sparse matrix.
    :param listidx: list of index to sample from (default None: it samples from
                    all shape[0] indexes).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """
    listidx = numpy.arange(Nent)
    listidx = listidx[numpy.random.permutation(len(listidx))]
    idx_term = 0
    arr = []
    for idx_ex in range(shape):
        if idx_term == len(listidx):
            idx_term = 0
        arr += [listidx[idx_term]]
        idx_term += 1
    return numpy.asarray(arr, dtype='int32')

def buildTriplesNeg(trainln,trainrn,traino):
    train_lex_neg=[]
    labelsTrain_neg = []
    for m in range(len(traino)):
        train_lex_neg += [[trainln[m],traino[m]]]
        labelsTrain_neg += [trainrn[m]]
    return train_lex_neg, labelsTrain_neg

def buildTriplesForward(trainl, trainr, traino):     
    train_lex=[]
    labels = []
    for m in range(len(traino)):
        train_lex += [[trainl[m],traino[m]]]
        labels += [trainr[m]]
    return train_lex, labels

def buildTriplesBackward(trainl, trainr, traino):     
    train_lex=[]
    labels = []
    for m in range(len(traino)):
        train_lex += [[trainr[m],traino[m]]]
        labels += [trainl[m]]
    return train_lex, labels


def FilteredRankingScoreIdx(sl, sr, idxl, idxr, idxo, true_triples):
    """
    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        il=numpy.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=numpy.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=numpy.argwhere(true_triples[:,2]==r).reshape(-1,)
 
        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i,0] for i in inter_l if true_triples[i,0] != l]
        scores_l = (sl(r, o)).flatten()
        scores_l[rmv_idx_l] = -numpy.inf
        errl += [numpy.argsort(numpy.argsort(-scores_l)).flatten()[l] + 1]

        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]
        scores_r = (sr(l, o)).flatten()
        scores_r[rmv_idx_r] = -numpy.inf
        errr += [numpy.argsort(numpy.argsort(-scores_r)).flatten()[r] + 1]
    return errl, errr


def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def build_matrices(alt_paths, st_node, rel_node, end_node, altrel2idx):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    ent_with_altPaths = alt_paths.keys()
    filt_list = []
    list_idx=[]
    cont=0
    for st, rel, end in zip(st_node, rel_node, end_node):
        if st in ent_with_altPaths and end in alt_paths[st]:
            for alp in alt_paths[st][end]:
                if rel not in alp[1:3]:
                    filt_list += [alp]
                    list_idx += [altrel2idx['%s: %s'%(rel, alp[1:3])]]
        cont +=1
    if len(filt_list)>0:
        tam=len(filt_list[0])
    else:
        return [],[],[]    
    x = numpy.zeros((tam, len(filt_list))).astype('int64')
    x_mask = numpy.zeros((tam-1, len(filt_list))).astype(theano.config.floatX)
    for idx, seq in enumerate(filt_list):
        x[:, idx] = seq
        x_mask[:, idx] = 1.
    return x, x_mask, numpy.asarray(list_idx)


def linear(x):
    return x
