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
from collections import OrderedDict


# Similarity functions -------------------------------------------------------
def L1sim(left, right):
    return - T.sum(T.abs_(left - right), axis=1)


def L2sim(left, right):
    return - T.sqrt(T.sum(T.sqr(left - right), axis=1))


def Dotsim(left, right):
    return T.sum(left * right, axis=1)
# -----------------------------------------------------------------------------


# Cost ------------------------------------------------------------------------
def margincost(pos, neg, marge=1.0):
    out = neg - pos + marge
    return T.sum(out * (out > 0)), out > 0
# -----------------------------------------------------------------------------


# Activation functions --------------------------------------------------------
def rect(x):
    return x * (x > 0)


def sigm(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def lin(x):
    return x
# -----------------------------------------------------------------------------


# Layers ----------------------------------------------------------------------
class Layer(object):
    """Class for a layer with one input vector w/o biases."""

    def __init__(self, rng, act, n_inp, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: name of the activation function ('lin', 'rect', 'tanh' or
                    'sigm').
        :param n_inp: input dimension.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
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
        """Forward function."""
        return self.act(T.dot(x, self.W))


class LayerLinear(object):
    """Class for a layer with two inputs vectors with biases."""

    def __init__(self, rng, act, n_inpl, n_inpr, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: name of the activation function ('lin', 'rect', 'tanh' or
                    'sigm').
        :param n_inpl: dimension of the 'left' input.
        :param n_inpr: dimension of the 'right' input.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
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
        """Forward function."""
        return self.act(self.layerl(x) + self.layerr(y) + self.b)


class LayerBilinear(object):
    """
    Class for a layer with bilinear interaction (n-mode vector-tensor product)
    on two input vectors with a tensor of parameters.
    """

    def __init__(self, rng, act, n_inpl, n_inpr, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: name of the activation function ('lin', 'rect', 'tanh' or
                    'sigm').
        :param n_inpl: dimension of the 'left' input.
        :param n_inpr: dimension of the 'right' input.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
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
        """Forward function."""
        xW = T.tensordot(x, self.W, axes=([1], [0]))
        xWy = ((y.reshape((y.shape[0], y.shape[1], 1))) * xW).sum(1)
        return self.act(xWy + self.b)

    def forwardrankrel(self, x, y):
        """Forward function in the special case of relation ranking to avoid a
        broadcast problem. @TODO: think about a workaround."""
        xW = T.tensordot(x, self.W, axes=([1], [0]))
        xW = xW.reshape((1, xW.shape[1], xW.shape[2]))
        xWy = ((y.reshape((y.shape[0], y.shape[1], 1))) * xW).sum(1)
        return self.act(xWy + self.b)


class LayerMat(object):
    """
    Class for a layer with two input vectors, the 'right' member being a flat
    representation of a matrix on which to perform the dot product with the
    'left' vector [Structured Embeddings model, Bordes et al. AAAI 2011].
    """

    def __init__(self, act, n_inp, n_out):
        """
        Constructor.

        :param act: name of the activation function ('lin', 'rect', 'tanh' or
                    'sigm').
        :param n_inp: input dimension.
        :param n_out: output dimension.

        :note: there is no parameter declared in this layer, the parameters
               are the embeddings of the 'right' member, therefore their
               dimension have to fit with those declared here: n_inp * n_out.
        """
        self.act = eval(act)
        self.actstr = act
        self.n_inp = n_inp
        self.n_out = n_out
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        # More details on the class and constructor comments.
        ry = y.reshape((y.shape[0], self.n_inp, self.n_out))
        rx = x.reshape((x.shape[0], x.shape[1], 1))
        return self.act((rx * ry).sum(1))


class LayerTrans(object):
    """
    Class for a layer with two input vectors that performs the sum of 
    of the 'left member' and 'right member'i.e. translating x by y.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x+y

class LayerdMat(object):
    """
    
    """

    def __init__(self):
        """
        Constructor.

        :note: there is no parameter declared in this layer, the parameters
               are the embeddings of the 'right' member, therefore their
               dimension have to fit with those declared here: n_inp * n_out.
        """
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        # More details on the class and constructor comments.

        return x * y

class Unstructured(object):
    """
    Class for a layer with two input vectors that performs the linear operator
    of the 'left member'.

    :note: The 'right' member is the relation, therefore this class allows to
    define an unstructured layer (no effect of the relation) in the same
    framework.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x

# ----------------------------------------------------------------------------


# Embeddings class -----------------------------------------------------------
class Embeddings(object):
    """Class for the embeddings matrix."""

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param N: number of entities, relations or both.
        :param D: dimension of the embeddings.
        :param tag: name of the embeddings for parameter declaration.
        """
        self.N = N
        self.D = D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(D, N))
        W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=0))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # embedding vectors.
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)
# ----------------------------------------------------------------------------


def parse_embeddings(embeddings):
    """
    Utilitary function to parse the embeddings parameter in a normalized way
    for the Structured Embedding [Bordes et al., AAAI 2011] and the Semantic
    Matching Energy [Bordes et al., AISTATS 2012] models.
    """
    if type(embeddings) == list:
        embedding = embeddings[0]
        relationl = embeddings[1]
        relationr = embeddings[2]
    else:
        embedding = embeddings
        relationl = embeddings
        relationr = embeddings
    return embedding, relationl, relationr


# Theano functions creation --------------------------------------------------
def SimFn(fnsim, embeddings, leftop, rightop):
    """
    This function returns a Theano function to measure the similarity score
    for sparse matrices inputs.

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr = S.csr_matrix('inpr')
    inpl = S.csr_matrix('inpl')
    inpo = S.csr_matrix('inpo')
    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    """
    Theano function inputs.
    :input inpl: sparse csr matrix (representing the indexes of the 'left'
                    entities), shape=(#examples, N [Embeddings]).
    :input inpr: sparse csr matrix (representing the indexes of the 'right'
                    entities), shape=(#examples, N [Embeddings]).
    :input inpo: sparse csr matrix (representing the indexes of the
                    relation member), shape=(#examples, N [Embeddings]).

    Theano function output
    :output simi: matrix of score values.
    """
    return theano.function([inpl, inpr, inpo], [simi],
            on_unused_input='ignore')


def RankRightFn(fnsim, embeddings, leftop, rightop,
                subtensorspec=None, adding=False):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of 'left' and relation members (as
    sparse matrices).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities)
    :param adding: if the right member is composed of several entities the
                   function needs to more inputs: we have to add the embedding
                   value of the other entities (with the appropriate scaling
                   factor to perform the mean pooling).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpl = S.csr_matrix('inpl')
    inpo = S.csr_matrix('inpo')
    if adding:
        inpradd = S.csr_matrix('inpradd')
        scal = T.scalar('scal')
    # Graph
    if subtensorspec is None:
        rhs = embedding.E.T
    else:
        # We compute the score only for a subset of entities
        rhs = embedding.E[:, :subtensorspec].T
    if adding:
        # Add the embeddings of the other entities (mean pooling)
        rhs = rhs * scal + (S.dot(embedding.E, inpradd).T).reshape(
                (1, embedding.D))
    lhs = (S.dot(embedding.E, inpl).T).reshape((1, embedding.D))
    rell = (S.dot(relationl.E, inpo).T).reshape((1, relationl.D))
    relr = (S.dot(relationr.E, inpo).T).reshape((1, relationr.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    """
    Theano function inputs.
    :input inpl: sparse csr matrix representing the indexes of the 'left'
                 entities, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the relation
                 member, shape=(#examples,N [Embeddings]).
    :opt input inpradd: sparse csr matrix representing the indexes of the
                        other entities of the 'right' member with the
                        appropriate scaling factor, shape = (#examples, N
                        [Embeddings]).
    :opt input scal: scaling factor to perform the mean: 1 / [#entities in the
                     member].

    Theano function output.
    :output simi: matrix of score values.
    """
    if not adding:
        return theano.function([inpl, inpo], [simi], on_unused_input='ignore')
    else:
        return theano.function([inpl, inpo, inpradd, scal], [simi],
                on_unused_input='ignore')


def RankLeftFn(fnsim, embeddings, leftop, rightop,
               subtensorspec=None, adding=False):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of 'right' and relation members (as
    sparse matrices).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities)
    :param adding: if the right member is composed of several entities the
                   function needs to more inputs: we have to add the embedding
                   value of the other entities (with the appropriate scaling
                   factor to perform the mean pooling).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr = S.csr_matrix('inpr')
    inpo = S.csr_matrix('inpo')
    if adding:
        inpladd = S.csr_matrix('inpradd')
        scal = T.scalar('scal')
    # Graph
    if subtensorspec is None:
        lhs = embedding.E.T
    else:
        # We compute the score only for a subset of entities
        lhs = embedding.E[:, :subtensorspec].T
    if adding:
        # Add the embeddings of the other entities (mean pooling)
        lhs = lhs * scal + (S.dot(embedding.E, inpladd).T).reshape(
                (1, embedding.D))
    rhs = (S.dot(embedding.E, inpr).T).reshape((1, embedding.D))
    rell = (S.dot(relationl.E, inpo).T).reshape((1, relationl.D))
    relr = (S.dot(relationr.E, inpo).T).reshape((1, relationr.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    """
    Theano function inputs.
    :input inpr: sparse csr matrix representing the indexes of the 'right'
                 entities, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the relation
                 member, shape=(#examples,N [Embeddings]).
    :opt input inpladd: sparse csr matrix representing the indexes of the
                        other entities of the 'left' member with the
                        appropriate scaling factor, shape = (#examples, N
                        [Embeddings]).
    :opt input scal: scaling factor to perform the mean: 1 / [#entities in the
                     member].

    Theano function output.
    :output simi: matrix of score values.
    """
    if not adding:
        return theano.function([inpr, inpo], [simi], on_unused_input='ignore')
    else:
        return theano.function([inpr, inpo, inpladd, scal], [simi],
                on_unused_input='ignore')


def RankRelFn(fnsim, embeddings, leftop, rightop,
              subtensorspec=None, adding=False):
    """
    This function returns a Theano function to measure the similarity score of
    all relation entities given couples of 'right' and 'left' entities (as
    sparse matrices).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities)
    :param adding: if the right member is composed of several entities the
                   function needs to more inputs: we have to add the embedding
                   value of the other entities (with the appropriate scaling
                   factor to perform the mean pooling).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr = S.csr_matrix('inpr')
    inpl = S.csr_matrix('inpl')
    if adding:
        inpoadd = S.csr_matrix('inpoadd')
        scal = T.scalar('scal')
    # Graph
    if subtensorspec is None:
        rell = relationl.E
        relr = relationr.E
    else:
        # We compute the score only for a subset of entities
        rell = relationl.E[:, :subtensorspec].T
        relr = relationr.E[:, :subtensorspec].T
    if adding:
        # Add the embeddings of the other entities (mean pooling)
        rell = rell * scal + (S.dot(relationl.E, inpoadd).T).reshape(
                (1, embedding.D))
        relr = relr * scal + (S.dot(relationr.E, inpoadd).T).reshape(
                (1, embedding.D))
    lhs = (S.dot(embedding.E, inpl).T).reshape((1, embedding.D))
    rhs = (S.dot(embedding.E, inpr).T).reshape((1, embedding.D))
    # hack to prevent a broadcast problem with the Bilinear layer
    if hasattr(leftop, 'forwardrankrel'):
        tmpleft = leftop.forwardrankrel(lhs, rell)
    else:
        tmpleft = leftop(lhs, rell)
    if hasattr(rightop, 'forwardrankrel'):
        tmpright = rightop.forwardrankrel(rhs, relr)
    else:
        tmpright = rightop(lhs, rell)
    simi = fnsim(tmpleft, tmpright)
    """
    Theano function inputs.
    :input inpl: sparse csr matrix representing the indexes of the 'left'
                 entities, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the 'right'
                 entities, shape=(#examples,N [Embeddings]).
    :opt input inpoadd: sparse csr matrix representing the indexes of the
                        other entities of the relation member with the
                        appropriate scaling factor, shape = (#examples, N
                        [Embeddings]).
    :opt input scal: scaling factor to perform the mean: 1 / [#entities in the
                     member].

    Theano function output.
    :output simi: matrix of score values.
    """
    if not adding:
        return theano.function([inpl, inpr], [simi], on_unused_input='ignore')
    else:
        return theano.function([inpl, inpr, inpoadd, scal], [simi],
                on_unused_input='ignore')


def SimFnIdx(fnsim, embeddings, leftop, rightop):
    """
    This function returns a Theano function to measure the similarity score
    for a given triplet of entity indexes.

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxo = T.iscalar('idxo')
    idxr = T.iscalar('idxr')
    idxl = T.iscalar('idxl')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: score value.
    """
    return theano.function([idxl, idxr, idxo], [simi],
            on_unused_input='ignore')


def RankRightFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxl = T.iscalar('idxl')
    idxo = T.iscalar('idxo')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    tmp = leftop(lhs, rell)
    simi = fnsim(tmp.reshape((1, tmp.shape[1])), rightop(rhs, relr))
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')


def RankRightFnIdxBi(embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[2]
    rel_matricesr = embeddings[3]
    
    # Inputs
    idxl = T.iscalar('idxl')
    idxo = T.iscalar('idxo')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    relmatricesl = (rel_matricesl.E[:, idxo]).reshape((1, rel_matricesl.D))
    relmatricesr = (rel_matricesr.E[:, idxo]).reshape((1, rel_matricesr.D))
    
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T

    W_m = W.E.reshape((1, W.D))
    
    expP = leftop(lhs, relmatricesl)+leftop(rhs, relmatricesr)+leftop(lhs, rightop(rhs, W_m))
    simi = -T.sum(expP, axis=1)
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')
    
    
def RankRightFnIdxTri(embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding = embeddings[0]
    rel_matrices = embeddings[1]
    
    # Inputs
    idxl = T.iscalar('idxl')
    idxo = T.iscalar('idxo')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    relmatrices = (rel_matrices.E[:, idxo]).reshape((1, rel_matrices.D))
    
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T
    
    expP = leftop(lhs, rightop(rhs, relmatrices))
    simi = -T.sum(expP, axis=1)
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')
    
    
def RankRightFnIdxTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embeddingbi = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[2]
    rel_matricesr = embeddings[3]
    embeddingtri = embeddings[4]
    rel_matricestri = embeddings[5]

    # Inputs
    idxl = T.iscalar('idxl')
    idxo = T.iscalar('idxo')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhsbi = (embeddingbi.E[:, :subtensorspec]).T
        rhstri = (embeddingtri.E[:, :subtensorspec]).T
    else:
        rhsbi = embeddingbi.E.T
        rhstri = embeddingtri.E.T

    lhsbi = (embeddingbi.E[:, idxl]).reshape((1, embeddingbi.D))
    lhstri = (embeddingtri.E[:, idxl]).reshape((1, embeddingtri.D))
    relmatricesl = (rel_matricesl.E[:, idxo]).reshape((1, rel_matricesl.D))
    relmatricesr = (rel_matricesr.E[:, idxo]).reshape((1, rel_matricesr.D))
    relmatricestri = (rel_matricestri.E[:, idxo]).reshape((1, rel_matricestri.D))


    W_m = W.E.reshape((1, W.D))

    expP = leftopbi(lhsbi, relmatricesl)+leftopbi(rhsbi, relmatricesr)+leftopbi(lhsbi, rightopbi(rhsbi, W_m)) + leftoptri(lhstri, rightoptri(rhstri, relmatricestri))
    simi = -T.sum(expP, axis=1)
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')
    
    
def RankLeftFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr = T.iscalar('idxr')
    idxo = T.iscalar('idxo')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    tmp = rightop(rhs, relr)
    simi = fnsim(leftop(lhs, rell), tmp.reshape((1, tmp.shape[1])))
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi],
            on_unused_input='ignore')


def RankLeftFnIdxBi(embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[2]
    rel_matricesr = embeddings[3]
    
    # Inputs
    idxr = T.iscalar('idxr')
    idxo = T.iscalar('idxo')
    # Graph
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    relmatricesl = (rel_matricesl.E[:, idxo]).reshape((1, rel_matricesl.D))
    relmatricesr = (rel_matricesr.E[:, idxo]).reshape((1, rel_matricesr.D))
    
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T

    W_m = W.E.reshape((1, W.D))

    expP = leftop(lhs, relmatricesl)+leftop(rhs, relmatricesr)+leftop(lhs, rightop(rhs, W_m))
    simi = -T.sum(expP, axis=1)
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi],
            on_unused_input='ignore')
            

def RankLeftFnIdxTri(embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding = embeddings[0]
    rel_matrices = embeddings[1]
    
    # Inputs
    idxr = T.iscalar('idxr')
    idxo = T.iscalar('idxo')
    # Graph
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    relmatrices = (rel_matrices.E[:, idxo]).reshape((1, rel_matrices.D))
    
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T

    expP = leftop(lhs, rightop(rhs, relmatrices))
    simi = -T.sum(expP, axis=1)
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi],
            on_unused_input='ignore')
            
            
def RankLeftFnIdxTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embeddingbi = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[2]
    rel_matricesr = embeddings[3]
    embeddingtri = embeddings[4]
    rel_matricestri = embeddings[5]
    
    # Inputs
    idxr = T.iscalar('idxr')
    idxo = T.iscalar('idxo')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhsbi = (embeddingbi.E[:, :subtensorspec]).T
        lhstri = (embeddingtri.E[:, :subtensorspec]).T
    else:
        lhsbi = embeddingbi.E.T
        lhstri = embeddingtri.E.T

    rhsbi = (embeddingbi.E[:, idxr]).reshape((1, embeddingbi.D))
    rhstri = (embeddingtri.E[:, idxr]).reshape((1, embeddingtri.D))
    relmatricesl = (rel_matricesl.E[:, idxo]).reshape((1, rel_matricesl.D))
    relmatricesr = (rel_matricesr.E[:, idxo]).reshape((1, rel_matricesr.D))
    relmatricestri = (rel_matricestri.E[:, idxo]).reshape((1, rel_matricestri.D))

    W_m = W.E.reshape((1, W.D))

    expP = leftopbi(lhsbi, relmatricesl)+leftopbi(rhsbi, relmatricesr)+leftopbi(lhsbi, rightopbi(rhsbi, W_m)) + leftoptri(lhstri, rightoptri(rhstri, relmatricestri))
    simi = -T.sum(expP, axis=1)
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi],
            on_unused_input='ignore')

def RankRelFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    embedding, relationl, relationr = parse_embeddings(embeddings)
    """
    This function returns a Theano function to measure the similarity score of
    all relation entities given couples of 'left' and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr = T.iscalar('idxo')
    idxl = T.iscalar('idxl')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rell = (relationl.E[:, :subtensorspec]).T
        relr = (relationr.E[:, :subtensorspec]).T
    else:
        rell = embedding.E.T
        relr = embedding.E.T
    # hack to prevent a broadcast problem with the Bilinear layer
    if hasattr(leftop, 'forwardrankrel'):
        tmpleft = leftop.forwardrankrel(lhs, rell)
    else:
        tmpleft = leftop(lhs, rell)
    if hasattr(rightop, 'forwardrankrel'):
        tmpright = rightop.forwardrankrel(rhs, relr)
    else:
        tmpright = rightop(lhs, rell)
    simi = fnsim(tmpleft, tmpright)
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxr: index value of the 'right' member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxr], [simi],
            on_unused_input='ignore')


def TrainFn(fnsim, embeddings, leftop, rightop, marge=1.0):
    """
    This function returns a theano function to perform a training iteration,
    contrasting couples of positive and negative triplets. members are given
    as sparse matrices. for one positive triplet there is one negative
    triplet.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge For the cost function.
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)
    # Inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    inpon = S.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')

    # Graph
    ## Positive triplet
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    ## Negative triplet
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    relln = S.dot(relationl.E, inpon).T
    relrn = S.dot(relationr.E, inpon).T
    simin = fnsim(leftop(lhsn, relln), rightop(rhsn, relrn))

    cost, out = margincost(simi, simin, marge)
    # Parameters gradients
    if hasattr(fnsim, 'params'):
        # If the similarity function has some parameters, we update them too.
        gradientsparams = T.grad(cost,
            leftop.params + rightop.params + fnsim.params)
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params + fnsim.params, gradientsparams))
    else:
        gradientsparams = T.grad(cost, leftop.params + rightop.params)
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params, gradientsparams))
    # Embeddings gradients
    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lrembeddings * gradients_embedding
    updates.update({embedding.E: newE})
    if type(embeddings) == list:
        # If there are different embeddings for the relation member.
        gradients_embedding = T.grad(cost, relationl.E)
        newE = relationl.E - lrparams * gradients_embedding
        updates.update({relationl.E: newE})
        gradients_embedding = T.grad(cost, relationr.E)
        newE = relationr.E - lrparams * gradients_embedding
        updates.update({relationr.E: newE})
    """
    Theano function inputs.
    :input lrembeddings: learning rate for the embeddings.
    :input lrparams: learning rate for the parameters.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpon: sparse csr matrix representing the indexes of the negative
                  triplet relation member, shape=(#examples,N [Embeddings]).

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function([lrembeddings, lrparams, inpl, inpr, inpo,
                           inpln, inprn, inpon],
                           [T.mean(cost), T.mean(out)], updates=updates,
                           on_unused_input='ignore')


def ForwardFn(fnsim, embeddings, leftop, rightop, marge=1.0):
    """
    This function returns a theano function to perform a forward step,
    contrasting couples of positive and negative triplets. members are given
    as sparse matrices. For one positive triplet there is one negative
    triplet.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.

    :note: this is useful for W_SABIE [Weston et al., IJCAI 2011]
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    inpon = S.csr_matrix()

    # graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    relln = S.dot(relationl.E, inpon).T
    relrn = S.dot(relationr.E, inpon).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    simin = fnsim(leftop(lhsn, relln), rightop(rhsn, relrn))
    cost, out = margincost(simi, simin, marge)
    """
    Theano function inputs.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpon: sparse csr matrix representing the indexes of the negative
                  triplet relation member, shape=(#examples,N [Embeddings]).

    Theano function output.
    :output out: binary vector representing when the margin is violated, i.e.
                 when an update occurs.
    """
    return theano.function([inpl, inpr, inpo,
                           inpln, inprn, inpon], [out],
                           on_unused_input='ignore')


def TrainFn1Member(fnsim, embeddings, leftop, rightop, marge=1.0, rel=True):
    """
    This function returns a theano function to perform a training iteration,
    contrasting positive and negative triplets. members are given as sparse
    matrices. For one positive triplet there are two or three (if rel == True)
    negative triplets. To create a negative triplet we replace only one member
    at a time.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')

    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    # Negative 'left' member
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr))
    # Negative 'right' member
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    cost = costl + costr
    out = T.concatenate([outl, outr])
    # List of inputs of the function
    list_in = [lrembeddings, lrparams,
            inpl, inpr, inpo, inpln, inprn]
    if rel:
        # If rel is True, we also consider a negative relation member
        inpon = S.csr_matrix()
        relln = S.dot(relationl.E, inpon).T
        relrn = S.dot(relationr.E, inpon).T
        simion = fnsim(leftop(lhs, relln), rightop(rhs, relrn))
        costo, outo = margincost(simi, simion, marge)
        cost += costo
        out = T.concatenate([out, outo])
        list_in += [inpon]

    if hasattr(fnsim, 'params'):
        # If the similarity function has some parameters, we update them too.
        gradientsparams = T.grad(cost,
            leftop.params + rightop.params + fnsim.params)
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params + fnsim.params, gradientsparams))
    else:
        gradientsparams = T.grad(cost, leftop.params + rightop.params)
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params, gradientsparams))
    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lrembeddings * gradients_embedding
    updates.update({embedding.E: newE})
    if type(embeddings) == list:
        # If there are different embeddings for the relation member.
        gradients_embedding = T.grad(cost, relationl.E)
        newE = relationl.E - lrparams * gradients_embedding
        updates.update({relationl.E: newE})
        gradients_embedding = T.grad(cost, relationr.E)
        newE = relationr.E - lrparams * gradients_embedding
        updates.update({relationr.E: newE})
    """
    Theano function inputs.
    :input lrembeddings: learning rate for the embeddings.
    :input lrparams: learning rate for the parameters.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :opt input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
            updates=updates, on_unused_input='ignore')


def TrainFn1MemberBi(embeddings, leftop, rightop, marge=1.0):
    """
    This function returns a theano function to perform a training iteration,
    contrasting positive and negative triplets. members are given as sparse
    matrices. For one positive triplet there are two or three (if rel == True)
    negative triplets. To create a negative triplet we replace only one member
    at a time.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.
    """
    embedding = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[2]
    rel_matricesr = embeddings[3]
    
    # Inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    lr = T.scalar('lr')
    lrparam = T.scalar('lrparam')

    ## Positive triplet
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    relmatricesl = S.dot(rel_matricesl.E, inpo).T
    relmatricesr = S.dot(rel_matricesr.E, inpo).T

    ## Negative triplet
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    relmatricesln = S.dot(rel_matricesl.E, inpo).T
    relmatricesrn = S.dot(rel_matricesr.E, inpo).T

    W_m = W.E.reshape((1, W.D))
    
    # Positive triplet score
    expP = leftop(lhs, relmatricesl)+leftop(rhs, relmatricesr)+leftop(lhs, rightop(rhs, W_m))
    simi = -T.sum(expP, axis=1)
    
    # Negative triple score
    expNl = leftop(lhsn, relmatricesl)+leftop(rhs, relmatricesr)+leftop(lhsn, rightop(rhs, W_m))
    expNr = leftop(lhs, relmatricesl)+leftop(rhsn, relmatricesr)+leftop(lhs, rightop(rhsn, W_m))
    similn = -T.sum(expNl, axis=1)
    simirn = -T.sum(expNr, axis=1)

    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)

    cost = costl + costr
    out = T.concatenate([outl, outr])

    # List of inputs of the function
    list_in = [lr, lrparam, inpl, inpr, inpo, inpln, inprn]


    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    updates = dict((i, i - lr * j) for i, j in zip(
        leftop.params + rightop.params, gradientsparams))

    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lr * gradients_embedding
    updates.update({embedding.E: newE})
    gradients_embeddingrelMatl = T.grad(cost, rel_matricesl.E)
    newrelMatl = rel_matricesl.E - lr * gradients_embeddingrelMatl
    updates.update({rel_matricesl.E: newrelMatl})
    gradients_embeddingrelMatr = T.grad(cost, rel_matricesr.E)
    newrelMatr = rel_matricesr.E - lr * gradients_embeddingrelMatr
    updates.update({rel_matricesr.E: newrelMatr})
    gradients_embeddingW = T.grad(cost, W.E)
    newW = W.E - lr * gradients_embeddingW
    updates.update({W.E: newW})
    
    """
    Theano function inputs.
    :input lrembeddings: learning rate for the embeddings.
    :input lrparams: learning rate for the parameters.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :opt input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
                updates=updates, on_unused_input='ignore')


def TrainFn1MemberTri(embeddings, leftop, rightop, marge=1.0):
    """
    This function returns a theano function to perform a training iteration,
    contrasting positive and negative triplets. members are given as sparse
    matrices. For one positive triplet there are two or three (if rel == True)
    negative triplets. To create a negative triplet we replace only one member
    at a time.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.
    """
    embedding = embeddings[0]
    rel_matrices = embeddings[1]
    
    # Inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    inpon = S.csr_matrix()
    lrparam = T.scalar('lrparam')
    lr = T.scalar('lr')

    ## Positive triplet
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    relmatrices = S.dot(rel_matrices.E, inpo).T

    ## Negative triplet
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    relmatricesn = S.dot(rel_matrices.E, inpo).T

    
    # Positive triplet score
    expP = leftop(lhs, rightop(rhs, relmatrices))
    simi = -T.sum(expP, axis=1)
    
    # Negative triple score
    expNl = leftop(lhsn, rightop(rhs, relmatrices))
    expNr = leftop(lhs, rightop(rhsn, relmatrices))
    similn = -T.sum(expNl, axis=1)
    simirn = -T.sum(expNr, axis=1)

    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)

    cost = costl + costr
    out = T.concatenate([outl, outr])

    # List of inputs of the function
    list_in = [lr, lrparam, inpl, inpr, inpo, inpln, inprn]


    gradientsparams = T.grad(cost, leftop.params + rightop.params)
    updates = dict((i, i - lr * j) for i, j in zip(
        leftop.params + rightop.params, gradientsparams))

    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lr * gradients_embedding
    updates.update({embedding.E: newE})
    gradients_embeddingrelMat = T.grad(cost, rel_matrices.E)
    newrelMat = rel_matrices.E - lr * gradients_embeddingrelMat
    updates.update({rel_matrices.E: newrelMat})
    
    """
    Theano function inputs.
    :input lrembeddings: learning rate for the embeddings.
    :input lrparams: learning rate for the parameters.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :opt input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
                updates=updates, on_unused_input='ignore')
                
                
def TrainFn1MemberTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri, marge=1.0):
    """
    This function returns a theano function to perform a training iteration,
    contrasting positive and negative triplets. members are given as sparse
    matrices. For one positive triplet there are two or three (if rel == True)
    negative triplets. To create a negative triplet we replace only one member
    at a time.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.
    """
    embeddingbi = embeddings[0]
    W = embeddings[1]
    rel_matricesl = embeddings[2]
    rel_matricesr = embeddings[3]
    embeddingtri = embeddings[4]
    rel_matricestri = embeddings[5]
    
    # Inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    lrbi = T.scalar('lrbi')
    lrtri = T.scalar('lrtri')
    
    ## Positive triplet
    lhsbi = S.dot(embeddingbi.E, inpl).T
    rhsbi = S.dot(embeddingbi.E, inpr).T
    lhstri = S.dot(embeddingtri.E, inpl).T
    rhstri = S.dot(embeddingtri.E, inpr).T
    relmatricesl = S.dot(rel_matricesl.E, inpo).T
    relmatricesr = S.dot(rel_matricesr.E, inpo).T
    reltri= S.dot(rel_matricestri.E, inpo).T

    ## Negative triplet
    lhsbin = S.dot(embeddingbi.E, inpln).T
    rhsbin = S.dot(embeddingbi.E, inprn).T
    lhstrin = S.dot(embeddingtri.E, inpln).T
    rhstrin = S.dot(embeddingtri.E, inprn).T
    relmatricesln = S.dot(rel_matricesl.E, inpo).T
    relmatricesrn = S.dot(rel_matricesr.E, inpo).T
    reltrin= S.dot(rel_matricestri.E, inpo).T

    W_m = W.E.reshape((1, W.D))
    
    # Positive triplet score
    expP = leftopbi(lhsbi, relmatricesl)+leftopbi(rhsbi, relmatricesr)+leftopbi(lhsbi, rightopbi(rhsbi, W_m))+ leftoptri(lhstri, rightoptri(rhstri, reltri))
    simi = -T.sum(expP, axis=1)
    
    # Negative triple score
    expNl = leftopbi(lhsbin, relmatricesl)+leftopbi(rhsbi, relmatricesr)+leftopbi(lhsbin, rightopbi(rhsbi, W_m))+ leftoptri(lhstrin, rightoptri(rhstri, reltri))
    expNr = leftopbi(lhsbi, relmatricesl)+leftopbi(rhsbin, relmatricesr)+leftopbi(lhsbi, rightopbi(rhsbin, W_m))+ leftoptri(lhstri, rightoptri(rhstrin, reltri))
    similn = -T.sum(expNl, axis=1)
    simirn = -T.sum(expNr, axis=1)

    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)

    cost = costl + costr
    out = T.concatenate([outl, outr]) 

    # List of inputs of the function
    list_in = [lrbi, lrtri, inpl, inpr, inpo, inpln, inprn]


    gradientsparams = T.grad(cost, leftopbi.params + rightopbi.params + leftoptri.params + rightoptri.params)
    updates = dict((i, i - lrbi * j) for i, j in zip(
        leftopbi.params + rightopbi.params + leftoptri.params + rightoptri.params, gradientsparams))

    gradients_embeddingbi = T.grad(cost, embeddingbi.E)
    newEbi = embeddingbi.E - lrbi * gradients_embeddingbi
    updates.update({embeddingbi.E: newEbi})
    gradients_embeddingtri = T.grad(cost, embeddingtri.E)
    newEtri = embeddingtri.E - lrtri * gradients_embeddingtri
    updates.update({embeddingtri.E: newEtri})
    gradients_embeddingrelMatl = T.grad(cost, rel_matricesl.E)
    newrelMatl = rel_matricesl.E - lrbi * gradients_embeddingrelMatl
    updates.update({rel_matricesl.E: newrelMatl})
    gradients_embeddingrelMatr = T.grad(cost, rel_matricesr.E)
    newrelMatr = rel_matricesr.E - lrbi * gradients_embeddingrelMatr
    updates.update({rel_matricesr.E: newrelMatr})
    gradients_embeddingrelMattri = T.grad(cost, rel_matricestri.E)
    newrelMattri = rel_matricestri.E - lrtri * gradients_embeddingrelMattri
    updates.update({rel_matricestri.E: newrelMattri})
    gradients_embeddingW = T.grad(cost, W.E)
    newW = W.E - lrbi * gradients_embeddingW
    updates.update({W.E: newW})
    
    """
    Theano function inputs.
    :input lrembeddings: learning rate for the embeddings.
    :input lrparams: learning rate for the parameters.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :opt input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
                updates=updates, on_unused_input='ignore')
                

def ForwardFn1Member(fnsim, embeddings, leftop, rightop, marge=1.0, rel=True):
    """
    This function returns a theano function to perform a forward step,
    contrasting positive and negative triplets. members are given as sparse
    matrices. For one positive triplet there are two or three (if rel == True)
    negative triplets. To create a negative triplet we replace only one member
    at a time.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.

    :note: this is useful for W_SABIE [Weston et al., IJCAI 2011]
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()

    # graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr))
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    list_in = [inpl, inpr, inpo, inpln]
    list_out = [outl, outr]
    if rel:
        inpon = S.csr_matrix()
        relln = S.dot(relationl.E, inpon).T
        relrn = S.dot(relationr.E, inpon).T
        simion = fnsim(leftop(lhs, relln), rightop(rhs, relrn))
        costo, outo = margincost(simi, simion, marge)
        out = T.concatenate([outl, outr, outo])
        list_in += [inpon]
        list_out += [outo]
    """
    Theano function inputs.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :opt input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).

    Theano function output.
    :output outl: binary vector representing when the margin is violated, i.e.
                  when an update occurs, for the 'left' member.
    :output outr: binary vector representing when the margin is violated, i.e.
                  when an update occurs, for the 'right' member.
    :opt output outo: binary vector representing when the margin is violated,
                  i.e. when an update occurs, for the relation member.
    """
    return theano.function(list_in, list_out, on_unused_input='ignore')


def RankingScoreIdx(sl, sr, idxl, idxr, idxo):
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
        errl += [np.argsort(np.argsort((
            sl(r, o)[0]).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((
            sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]
    return errl, errr


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
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)
 
        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i,0] for i in inter_l if true_triples[i,0] != l]
        scores_l = (sl(r, o)[0]).flatten()
        scores_l[rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l)).flatten()[l] + 1]

        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]
        scores_r = (sr(l, o)[0]).flatten()
        scores_r[rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r)).flatten()[r] + 1]
    return errl, errr
    
    
def RankingScoreRightIdx(sr, idxl, idxr, idxo):
    """
    This function computes the rank list of the rhs, over a list of lhs, rhs
    and rel indexes.

    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        errr += [np.argsort(np.argsort((
            sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]
    return errr


def RankingScoreRelIdx(so, idxl, idxr, idxo):
    """
    This function computes the rank list of the rel, over a list of lhs, rhs
    and rel indexes.

    :param so: Theano function created with RankRelFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    erro = []
    for l, o, r in zip(idxl, idxo, idxr):
        erro += [np.argsort(np.argsort((
            so(l, r)[0]).flatten())[::-1]).flatten()[o] + 1]
    return erro


def RankingScore(sl, sr, so, inpl, inpr, inpo):
    """
    This function computes the rank list of the lhs, rhs and relation, with
    members given as sparse index matrices.

    :param sl: Theano function created with RankLeftFn().
    :param sr: Theano function created with RankRightFn().
    :param so: Theano function created with RankRelFn().
    :param inpl: sparse index matrix for the 'left' member.
    :param inpr: sparse index matrix for the 'right' member.
    :param inpo: sparse index matrix for the relation member.
    """
    errl = []
    errr = []
    erro = []
    for i in range(inpl.shape[1]):
        rankl = np.argsort((sl(inpr[:, i], inpo[:, i])[0]).flatten())
        for l in inpl[:, i].nonzero()[0]:
            errl += [np.argsort(rankl[::-1]).flatten()[l] + 1]
        rankr = np.argsort((sr(inpl[:, i], inpo[:, i])[0]).flatten())
        for r in inpr[:, i].nonzero()[0]:
            errr += [np.argsort(rankr[::-1]).flatten()[r] + 1]
        ranko = np.argsort((so(inpl[:, i], inpr[:, i])[0]).flatten())
        for o in inpo[:, i].nonzero()[0]:
            erro += [np.argsort(ranko[::-1]).flatten()[o] + 1]
    return errr, errl, erro


def RankingScoreWSD(sl, sr, so, inpl, inpr, inpo, inplc, inprc, inpoc):
    """
    This function computes the rank list of the lhs, rhs and relation, with
    members given as sparse index matrices. It replace only one word per
    member and does all combinations.

    :param sl: Theano function created with RankLeftFn() with adding == True.
    :param sr: Theano function created with RankRightFn() with adding == True.
    :param so: Theano function created with RankRelFn() with adding == True.
    :param inpl: sparse index matrix for the 'left' member.
    :param inpr: sparse index matrix for the 'right' member.
    :param inpo: sparse index matrix for the relation member.
    :param inplc: sparse matrix with the true 'left' member index
                  correspondance.
    :param inprc: sparse matrix with the true 'right' member index
                  correspondance.
    :param inpoc: sparse matrix with the true relation member index
                  correspondance.
    """
    errl = []
    errr = []
    erro = []
    for i in range(inpl.shape[1]):
        lnz = inpl[:, i].nonzero()[0]
        for j in lnz:
            val = inpl[j, i]
            tmpadd = copy.deepcopy(inpl[:, i])
            tmpadd[j, 0] = 0.0
            rankl = np.argsort((sl(inpr[:, i], inpo[:, i],
                                      tmpadd, val)[0]).flatten())
            errl += [np.argsort(rankl[::-1]).flatten()[inplc[j, i]] + 1]
        rnz = inpr[:, i].nonzero()[0]
        for j in rnz:
            val = inpr[j, i]
            tmpadd = copy.deepcopy(inpr[:, i])
            tmpadd[j, 0] = 0.0
            rankr = np.argsort((sr(inpl[:, i], inpo[:, i],
                                      tmpadd, val)[0]).flatten())
            errr += [np.argsort(rankr[::-1]).flatten()[inprc[j, i]] + 1]
        onz = inpo[:, i].nonzero()[0]
        for j in onz:
            val = inpo[j, i]
            tmpadd = copy.deepcopy(inpo[:, i])
            tmpadd[j, 0] = 0.0
            ranko = np.argsort((so(inpl[:, i], inpr[:, i],
                                      tmpadd, val)[0]).flatten())
            erro += [np.argsort(ranko[::-1]).flatten()[inpoc[j, i]] + 1]
    return errl, errr, erro
# ----------------------------------------------------------------------------
