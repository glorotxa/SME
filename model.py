import os
import sys
import time
import copy
import cPickle

import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T


# Similarity functions ----------------------------
def L1sim(left, right):
    return - T.sum(T.abs_(left - right), axis=1)


def L2sim(left, right):
    return - T.sqrt(T.sum(T.sqr(left - right), axis=1))


def Dotsim(left, right):
    return T.sum(left * right, axis=1)
# -------------------------------------------------


# Cost -------------------------------------------
def margincost(pos, neg, marge=1.0):
    out = neg - pos + marge
    return T.sum(out * (out > 0)), out > 0
# -------------------------------------------------


# Activation functions ----------------------------
def rect(x):
    return x * (x > 0)


def sigm(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def lin(x):
    return x
# -------------------------------------------------


# Layers ------------------------------------------
class Layer(object):
    def __init__(self, rng, act, n_inp, n_out, tag=''):
        self.act = eval(act)
        self.actstr = act
        self.n_inp = n_inp
        self.n_out = n_out
        wbound = np.sqrt(6. / (n_inp + n_out))
        W_values = np.asarray(
            rng.uniform(low=-wbound, high=wbound, size=(n_inp, n_out)),
            dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W' + tag)
        self.params = [self.W]

    def __call__(self, x):
        return self.act(T.dot(x, self.W))


class LayerLinear(object):
    def __init__(self, rng, act, n_inpl, n_inpr, n_out, tag=''):
        self.act = eval(act)
        self.actstr = act
        self.n_inpl = n_inpl
        self.n_inpr = n_inpr
        self.n_out = n_out
        self.layerl = Layer(rng, 'lin', n_inpl, n_out, tag='left' + tag)
        self.layerr = Layer(rng, 'lin', n_inpr, n_out, tag='right' + tag)
        b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b' + tag)
        self.params = self.layerl.params + self.layerr.params + [self.b]

    def __call__(self, x, y):
        return self.act(self.layerl(x) + self.layerr(y) + self.b)


class LayerBilinear(object):
    def __init__(self, rng, act, n_inpl, n_inpr, n_out, tag=''):
        self.act = eval(act)
        self.actstr = act
        self.n_inpl = n_inpl
        self.n_inpr = n_inpr
        self.n_out = n_out
        wbound = np.sqrt(9. / (n_inpl + n_inpr + n_out))
        W_values = rng.uniform(low=-wbound, high=wbound,
                               size=(n_inpl, n_inpr, n_out))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W' + tag)
        b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b' + tag)
        self.params = [self.W, self.b]

    def __call__(self, x, y):
        xW = T.tensordot(x, self.W, axes=([1], [0]))
        xWy = ((y.reshape((y.shape[0], y.shape[1], 1))) * xW).sum(1)
        return self.act(xWy + self.b)


class LayerMat(object):
    def __init__(self, act, n_inp, n_out):
        self.act = eval(act)
        self.actstr = act
        self.n_inp = n_inp
        self.n_out = n_out
        self.params = []

    def __call__(self, x, y):
        ry = y.reshape((y.shape[0], self.n_inp, self.n_out))
        rx = x.reshape((x.shape[0], x.shape[1], 1))
        return self.act((rx * ry).sum(1))


class Unstructured(object):
    def __init__(self):
        self.params = []

    def __call__(self, x, y):
        return x
# ---------------------------------------


# Embeddings class ----------------------
class Embeddings(object):
    def __init__(self, rng, N, D, tag=''):
        self.N = N
        self.D = D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(D, N))
        W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=0))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        self.updates = {self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=0))}
        self.normalize = theano.function([], [], updates=self.updates)
# ---------------------------------------


def parse_embeddings(embeddings):
    if type(embeddings) == list:
        embedding = embeddings[0]
        relationl = embeddings[1]
        relationr = embeddings[2]
    else:
        embedding = embeddings
        relationl = embeddings
        relationr = embeddings
    return embedding, relationl, relationr


def SimFn(fnsim, embeddings, leftop, rightop):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # inputs
    inpposr = S.csr_matrix()
    inpposl = S.csr_matrix()
    inpposo = S.csr_matrix()
    # graph
    lhs = S.dot(embedding.E, inpposl).T
    rhs = S.dot(embedding.E, inpposr).T
    rell = S.dot(relationl.E, inpposo).T
    relr = S.dot(relationr.E, inpposo).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    return theano.function([inpposl, inpposr, inpposo], [simi],
            on_unused_input='ignore')


def RankRightFn(fnsim, embeddings, leftop, rightop,
                subtensorspec=None, adding=False):
    # Scoring fuynction with respect to the complete list of embeddings (or a
    # subtensor of it defined by subtensorspec) if adding = True the scoring
    # function has 2 more arguments
    embedding, relationl, relationr = parse_embeddings(embeddings)

    idxrel = S.csr_matrix('idxrel')
    idxleft = S.csr_matrix('idxleft')
    lhs = (S.dot(embedding.E, idxleft).T).reshape((1, embedding.D))
    if not adding:
        if subtensorspec is None:
            rhs = embedding.E.T
        else:
            rhs = embedding.E[:, :subtensorspec].T
    else:
        idxadd = S.csr_matrix('idxadd')
        sc = T.scalar('sc')
        if subtensorspec is None:
            rhs = embedding.E.T * sc + (S.dot(embedding.E,
                idxadd).T).reshape((1, embedding.D))
        else:
            rhs = embedding.E[:, :subtensorspec].T * sc + (S.dot(
                embedding.E, idxadd).T).reshape((1, embedding.D))
    rell = (S.dot(relationl.E, idxrel).T).reshape((1, relationl.D))
    relr = (S.dot(relationr.E, idxrel).T).reshape((1, relationr.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    if not adding:
        return theano.function([idxleft, idxrel], [simi],
                on_unused_input='ignore')
    else:
        return theano.function([idxleft, idxrel, idxadd, sc], [simi],
                on_unused_input='ignore')


# Ask for the left member
def RankLeftFn(fnsim, embeddings, leftop, rightop,
               subtensorspec=None, adding=False):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    idxrel = S.csr_matrix('idxrel')
    idxright = S.csr_matrix('idxright')
    rhs = (S.dot(embedding.E,
        idxright).T).reshape((1, embedding.D))
    if not adding:
        if subtensorspec is None:
            lhs = embedding.E.T
        else:
            lhs = embedding.E[:, :subtensorspec].T
    else:
        idxadd = S.csr_matrix('idxadd')
        sc = T.scalar('sc')
        if subtensorspec is None:
            lhs = embedding.E.T * sc + (S.dot(embedding.E,
                idxadd).T).reshape((1, embedding.D))
        else:
            lhs = embedding.E[:, :subtensorspec].T * sc + (S.dot(
                embedding.E, idxadd).T).reshape((1, embedding.D))
    rell = (S.dot(relationl.E, idxrel).T).reshape((1, relationl.D))
    relr = (S.dot(relationr.E, idxrel).T).reshape((1, relationr.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    if not adding:
        return theano.function([idxright, idxrel], [simi],
                on_unused_input='ignore')
    else:
        return theano.function([idxright, idxrel, idxadd, sc], [simi],
                on_unused_input='ignore')


# Ask for the relation member
def RankRelFn(fnsim, embeddings, leftop, rightop,
              subtensorspec=None, adding=False):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    idxright = S.csr_matrix('idxright')
    idxleft = S.csr_matrix('idxleft')
    lhs = (S.dot(embedding.E,
        idxleft).T).reshape((1, embedding.D))
    if not adding:
        if subtensorspec is None:
            rell = relationl.E.T
            relr = relationr.E.T
        else:
            rell = relationl.E[:, :subtensorspec].T
            relr = relationr.E[:, :subtensorspec].T
    else:
        idxadd = S.csr_matrix('idxadd')
        sc = T.scalar('sc')
        if subtensorspec is None:
            rell = relationl.E.T * sc + (S.dot(relationl.E,
                idxadd).T).reshape((1, relationl.D))
            relr = relationr.E.T * sc + (S.dot(relationr.E,
                idxadd).T).reshape((1, relationr.D))
        else:
            rell = relationl.E[:, :subtensorspec].T * sc + (S.dot(
                relationl.E, idxadd).T).reshape((1, relationl.D))
            relr = relationr.E[:, :subtensorspec].T * sc + (S.dot(
                relationr.E, idxadd).T).reshape((1, relationr.D))
    rhs = (S.dot(embedding.E,
        idxright).T).reshape((1, embedding.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    if not adding:
        return theano.function([idxleft, idxright], [simi],
                on_unused_input='ignore')
    else:
        return theano.function([idxleft, idxright, idxadd, sc], [simi],
                on_unused_input='ignore')


# Creation of scoring function on indexes (not on sparse matrices)
def SimFnIdx(fnsim, embeddings, leftop, rightop):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxright')
    idxleft = T.iscalar('idxleft')
    lhs = (embedding.E[:, idxleft]).reshape((1, embedding.D))
    rhs = (embedding.E[:, idxright]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxrel]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxrel]).reshape((1, relationr.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    return theano.function([idxleft, idxright, idxrel], [simi],
            on_unused_input='ignore')


# Ask for the right member
def RankRightFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    idxrel = T.iscalar('idxrel')
    idxleft = T.iscalar('idxleft')
    lhs = (embedding.E[:, idxleft]).reshape((1, embedding.D))
    if subtensorspec is not None:
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T
    rell = (relationl.E[:, idxrel]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxrel]).reshape((1, relationr.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    return theano.function([idxleft, idxrel], [simi], on_unused_input='ignore')


# Ask for the left member
def RankLeftFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    idxrel = T.iscalar('idxrel')
    idxright = T.iscalar('idxright')
    rhs = (embedding.E[:, idxright]).reshape((1, embedding.D))
    if subtensorspec is not None:
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T
    rell = (relationl.E[:, idxrel]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxrel]).reshape((1, relationr.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    return theano.function([idxright, idxrel], [simi],
            on_unused_input='ignore')


# Ask for the relation member
def RankRelFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    idxright = T.iscalar('idxrel')
    idxleft = T.iscalar('idxleft')
    lhs = (embedding.E[:, idxleft]).reshape((1, embedding.D))
    if subtensorspec is not None:
        rell = (relationl.E[:, :subtensorspec]).T
        relr = (relationr.E[:, :subtensorspec]).T
    else:
        rell = embedding.E.T
        relr = embedding.E.T
    rhs = (embedding.E[:, idxright]).reshape((1, embedding.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    return theano.function([idxleft, idxright], [simi],
            on_unused_input='ignore')


# The training function creation:
# rel = true, negative sample for the relation member.
# lrparams = learning rate for all the parameters of the model.
# lrembeddings = learning rate for the embeddings.
# inpposl = sparse matrix of the lhs.
# inposr = sparse matrix of the rhs
# inposo = sparse matrix of the relation
# inpposln = sparse matrix of the negatif samples for the lhs
# inpposrn = sparse matrix of the negatif samples for the rhs
# inpposon = sparse matrix of the negatif samples for the relation
def TrainFn(fnsim, embeddings, leftop, rightop, marge=1.0):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # inputs
    inpposr = S.csr_matrix()
    inpposl = S.csr_matrix()
    inpposo = S.csr_matrix()
    inpposln = S.csr_matrix()
    inpposrn = S.csr_matrix()
    inpposon = S.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')

    # graph
    lhs = S.dot(embedding.E, inpposl).T
    rhs = S.dot(embedding.E, inpposr).T
    rell = S.dot(relationl.E, inpposo).T
    relr = S.dot(relationr.E, inpposo).T
    lhsn = S.dot(embedding.E, inpposln).T
    rhsn = S.dot(embedding.E, inpposrn).T
    relln = S.dot(relationl.E, inpposon).T
    relrn = S.dot(relationr.E, inpposon).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    simin = fnsim(leftop(lhsn, relln), rightop(rhsn, relrn))
    cost, out = margincost(simi, simin, marge)
    if hasattr(fnsim, 'params'):
        gradientsparams = T.grad(cost,
            leftop.params + rightop.params + fnsim.params)
        updates = dict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params + fnsim.params, gradientsparams))
    else:
        gradientsparams = T.grad(cost, leftop.params + rightop.params)
        updates = dict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params, gradientsparams))
    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lrembeddings * gradients_embedding
    updates.update({embedding.E: newE})
    if type(embeddings) == list:
        gradients_embedding = T.grad(cost, relationl.E)
        newE = relationl.E - lrparams * gradients_embedding
        updates.update({relationl.E: newE})
        gradients_embedding = T.grad(cost, relationr.E)
        newE = relationr.E - lrparams * gradients_embedding
        updates.update({relationr.E: newE})
    return theano.function([lrembeddings, lrparams, inpposl, inpposr, inpposo,
                           inpposln, inpposrn, inpposon],
                           [T.mean(cost), T.mean(out)], updates=updates,
                           on_unused_input='ignore')


# Function returning the binary vector representing: cost>0
def ForwardFn(fnsim, embeddings, leftop, rightop, marge=1.0):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # inputs
    inpposr = S.csr_matrix()
    inpposl = S.csr_matrix()
    inpposo = S.csr_matrix()
    inpposln = S.csr_matrix()
    inpposrn = S.csr_matrix()
    inpposon = S.csr_matrix()

    # graph
    lhs = S.dot(embedding.E, inpposl).T
    rhs = S.dot(embedding.E, inpposr).T
    rell = S.dot(relationl.E, inpposo).T
    relr = S.dot(relationr.E, inpposo).T
    lhsn = S.dot(embedding.E, inpposln).T
    rhsn = S.dot(embedding.E, inpposrn).T
    relln = S.dot(relationl.E, inpposon).T
    relrn = S.dot(relationr.E, inpposon).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    simin = fnsim(leftop(lhsn, relln), rightop(rhsn, relrn))
    cost, out = margincost(simi, simin, marge)
    return theano.function([inpposl, inpposr, inpposo,
                           inpposln, inpposrn, inpposon], [out],
                           on_unused_input='ignore')


def TrainFn1Member(fnsim, embeddings, leftop, rightop, marge=1.0, rel=True):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # inputs
    inpposr = S.csr_matrix()
    inpposl = S.csr_matrix()
    inpposo = S.csr_matrix()
    inpposln = S.csr_matrix()
    inpposrn = S.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')

    # graph
    lhs = S.dot(embedding.E, inpposl).T
    rhs = S.dot(embedding.E, inpposr).T
    rell = S.dot(relationl.E, inpposo).T
    relr = S.dot(relationr.E, inpposo).T
    lhsn = S.dot(embedding.E, inpposln).T
    rhsn = S.dot(embedding.E, inpposrn).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr))
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    cost = costl + costr
    out = T.concatenate([outl, outr])
    list_in = [lrembeddings, lrparams,
            inpposl, inpposr, inpposo, inpposln, inpposrn]
    if rel:
        inpposon = S.csr_matrix()
        relln = S.dot(relationl.E, inpposon).T
        relrn = S.dot(relationr.E, inpposon).T
        simion = fnsim(leftop(lhs, relln), rightop(rhs, relrn))
        costo, outo = margincost(simi, simion, marge)
        cost += costo
        out = T.concatenate([out, outo])
        list_in += [inpposon]

    if hasattr(fnsim, 'params'):
        gradientsparams = T.grad(cost,
            leftop.params + rightop.params + fnsim.params)
        updates = dict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params + fnsim.params, gradientsparams))
    else:
        gradientsparams = T.grad(cost, leftop.params + rightop.params)
        updates = dict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params, gradientsparams))
    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lrembeddings * gradients_embedding
    updates.update({embedding.E: newE})
    if type(embeddings) == list:
        gradients_embedding = T.grad(cost, relationl.E)
        newE = relationl.E - lrparams * gradients_embedding
        updates.update({relationl.E: newE})
        gradients_embedding = T.grad(cost, relationr.E)
        newE = relationr.E - lrparams * gradients_embedding
        updates.update({relationr.E: newE})
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
            updates=updates, on_unused_input='ignore')


# Function returning the binary vector representing: cost>0
def ForwardFn1Member(fnsim, embeddings, leftop, rightop, marge=1.0, rel=True):
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # inputs
    inpposr = S.csr_matrix()
    inpposl = S.csr_matrix()
    inpposo = S.csr_matrix()
    inpposln = S.csr_matrix()
    inpposrn = S.csr_matrix()

    # graph
    lhs = S.dot(embedding.E, inpposl).T
    rhs = S.dot(embedding.E, inpposr).T
    rell = S.dot(relationl.E, inpposo).T
    relr = S.dot(relationr.E, inpposo).T
    lhsn = S.dot(embedding.E, inpposln).T
    rhsn = S.dot(embedding.E, inpposrn).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr))
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    list_in = [inpposl, inpposr, inpposo, inpposln]
    list_out = [outl, outr]
    if rel:
        inpposon = S.csr_matrix()
        relln = S.dot(relationl.E, inpposon).T
        relrn = S.dot(relationr.E, inpposon).T
        simion = fnsim(leftop(lhs, relln), rightop(rhs, relrn))
        costo, outo = margincost(simi, simion, marge)
        out = T.concatenate([outl, outr, outo])
        list_in += [inpposon]
        list_out += [outo]
    return theano.function(list_in, list_out, on_unused_input='ignore')


# Compute the rank list of the lhs and rhs, over a list of lhs, rhs and rel
# indexes.  Only works when there is one word per member (WordNet) sl build
# with RankLeftFnIdx sr build with RankRightFnIdx
def RankingScoreIdx(sl, sr, idxtl, idxtr, idxto):
    errl = []
    errr = []
    for l, o, r in zip(idxtl, idxto, idxtr):
        errl += [np.argsort(np.argsort((
            sl(r, o)[0]).flatten())[::-1]).flatten()[l]]
        errr += [np.argsort(np.argsort((
            sr(l, o)[0]).flatten())[::-1]).flatten()[r]]
    return errl, errr


# Similar but works with sparse index matrices (posl,posr,poso) = (lhs,rhs,rel)
# replace the whole member by one word.  sl build with RankLeftFnIdx sr build
# with RankRightFnIdx so build with RankRelFnIdx
def RankingScore(sl, sr, so, posl, posr, poso):
    errl = []
    errr = []
    erro = []
    for i in range(posl.shape[1]):
        rankl = np.argsort((sl(posr[:, i], poso[:, i])[0]).flatten())
        for l in posl[:, i].nonzero()[0]:
            errl += [np.argsort(rankl[::-1]).flatten()[l]]
        rankr = np.argsort((sr(posl[:, i], poso[:, i])[0]).flatten())
        for r in posr[:, i].nonzero()[0]:
            errr += [np.argsort(rankr[::-1]).flatten()[r]]
        ranko = np.argsort((so(posl[:, i], posr[:, i])[0]).flatten())
        for o in poso[:, i].nonzero()[0]:
            erro += [np.argsort(ranko[::-1]).flatten()[0]]
    return errr, errl, erro


# The same : Similar but works with sparse index matrices (posl,posr,poso) =
# (lhs,rhs,rel) AND replace only ONE word per member (does ALL combinations) sl
# build with SimilarityFunctionleftl (with the adding argument = True) sr build
# with SimilarityFunctionrightl (with the adding argument = True) so build with
# SimilarityFunctionrell (with the adding argument = True) But compares with
# the index correspondance sparse matrices: (poslc,posrc,posoc) (you give
# lemmas in input and find the ranking of synsets).
def RankingScoreWSD(sl, sr, so, posl, posr, poso, poslc, posrc, posoc):
    errl = []
    errr = []
    erro = []
    for i in range(posl.shape[1]):
        lnz = posl[:, i].nonzero()[0]
        for j in lnz:
            val = posl[j, i]
            tmpadd = copy.deepcopy(posl[:, i])
            tmpadd[j, 0] = 0.0
            rankl = np.argsort((sl(posr[:, i], poso[:, i],
                                      tmpadd, val)[0]).flatten())
            errl += [np.argsort(rankl[::-1]).flatten()[poslc[j, i]]]
        rnz = posr[:, i].nonzero()[0]
        for j in rnz:
            val = posr[j, i]
            tmpadd = copy.deepcopy(posr[:, i])
            tmpadd[j, 0] = 0.0
            rankr = np.argsort((sr(posl[:, i], poso[:, i],
                                      tmpadd, val)[0]).flatten())
            errr += [np.argsort(rankr[::-1]).flatten()[posrc[j, i]]]
        onz = poso[:, i].nonzero()[0]
        for j in onz:
            val = poso[j, i]
            tmpadd = copy.deepcopy(poso[:, i])
            tmpadd[j, 0] = 0.0
            ranko = np.argsort((so(posl[:, i], posr[:, i],
                                      tmpadd, val)[0]).flatten())
            erro += [np.argsort(ranko[::-1]).flatten()[posoc[j, i]]]
    return errl, errr, erro
