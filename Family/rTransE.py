'''
Build a tweet sentiment analyzer
'''
from collections import OrderedDict
import cPickle as pkl
import random
import sys
import time

import scipy.io

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from aux_func import *

def L2sim(left, right):
    return - tensor.sqrt(tensor.sum(tensor.sqr(left - right), axis=1))

def L2norm(left, right):
    return tensor.sum(tensor.sqr(left - right), axis=1)

def margincost(pos, neg, marge=1.0):
    out = neg - pos + marge
    return tensor.sum(out * (out > 0)), out > 0


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)



def zipp(params, tparams):
    """
    When we reload the model. 
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    wb = numpy.sqrt(6. / options['dim_proj'])
    Wemb = numpy.random.uniform(low=-wb, high=wb, size=(options['n_words'], options['dim_proj']))
    Wemb = Wemb.T / numpy.sqrt(numpy.sum(Wemb ** 2, axis=1))
    params['Wemb'] = numpy.asarray(Wemb.T, dtype=config.floatX)
            
    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def rtranse_layer(tparams, state_below, options, prefix='rnn', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _step(m_, x_, h_):
        h_t = x_ + h_
        h_t = m_[:, None] * h_t + (1. - m_)[:, None] * h_
        return h_t


    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval

# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'rtranse':(rtranse_layer)}


def sgd(lr, tparams, grads, xP, xN, mask, yP, yN, x, y, maskAlt, cost, weight):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([xP, xN, mask, yP, yN, x, y, maskAlt, weight], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lrateEmb, lrateRNN, tparams, grads, xP, xN, mask, yP, yN, x, y, maskAlt, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()] 

    #They are not updated until the updates parameter is called.
    #See http://nbviewer.ipython.org/github/jaberg/IPythonTheanoTutorials/blob/master/ipynb/Theano%20Tutorial%20%28Part%203%20-%20Functions%20and%20Shared%20Variables%29.ipynb

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([xP, xN, mask, yP, yN, x, y, maskAlt], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lrateEmb, lrateRNN], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lrateEmb, lrateRNN, tparams, grads, xP, xN, mask, yP, yN, x, y, maskAlt, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([xP, xN, mask, yP, yN, x, y, maskAlt], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lrateEmb, lrateRNN], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(1234)

    xP = tensor.matrix('xP', dtype='int64')
    yP = tensor.vector('yP', dtype='int64')
    xN = tensor.matrix('xN', dtype='int64')
    yN = tensor.vector('yN', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)

    n_timesteps = mask.shape[0]
    n_samples = mask.shape[1]

    embP = tparams['Wemb'][xP.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    embN = tparams['Wemb'][xN.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    outP = tparams['Wemb'][yP.flatten()].reshape([n_samples,
                                                options['dim_proj']])
    outN = tparams['Wemb'][yN.flatten()].reshape([n_samples,
                                                options['dim_proj']])
    
    projP = get_layer(options['encoder'])(tparams, embP, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    projN = get_layer(options['encoder'])(tparams, embN, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    
    projP = projP[-1,:,:]
    projN = projN[-1,:,:]


    simi= L2sim(projP, outP)
    similn= L2sim(projN, outP)
    simirn= L2sim(projP, outN)
    coding_err = L2norm(projP, outP)
        
    costl, outl = margincost(simi, similn, options['margin'])
    costr, outr = margincost(simi, simirn, options['margin'])

    L2norms_sum = 0
    for sym in tparams.keys():
        L2norms_sum += tensor.sum(tensor.sqr(tparams[sym]))

    cost = costl + costr
    cost += options['lambda_pen']*(tensor.sum(coding_err)) + options['penalty_norm']*(L2norms_sum)

    return xP, xN, mask, yP, yN, cost

def autoencoding_errors(tparams, options, cost):
    x = tensor.matrix('x', dtype='int64')
    y = tensor.vector('y', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    weight = tensor.vector('weight', dtype=config.floatX)

    n_timesteps = mask.shape[0]
    n_samples = mask.shape[1]
    
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])

    out = tparams['Wemb'][y.flatten()].reshape([n_samples,
                                                options['dim_proj']])

    proj = get_layer(options['encoder'])(tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    proj = proj[-1,:,:]
    coding_err = L2norm(proj, out)    

    cost += tensor.sum(options['alpha']*weight*coding_err)

    return x, y, mask, weight, cost

    
def RankLeftFn(tparams, options):
    idxr=tensor.scalar('idxr', dtype='int64')
    idxo=tensor.scalar('idxo',dtype='int64')

    embL = tparams['Wemb'][tensor.arange(options['n_ent'])].reshape([1,
                                            options['n_ent'],
                                            options['dim_proj']])
    embO = tparams['Wemb'][idxo].reshape([1, options['dim_proj']])
    embO = tensor.tile(embO, (options['n_ent'],1))[None,:,:]
    emb=tensor.concatenate([embL, embO])
    out = tparams['Wemb'][idxr].reshape([1, options['dim_proj']])

    time_steps = emb.shape[0]
    n_samples = emb.shape[1]
    mask = tensor.alloc(numpy_floatX(1), time_steps, n_samples)
    proj = get_layer(options['encoder'])(tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    proj = proj[-1,:,:]
    simi= L2sim(proj, out)
        
    return theano.function([idxr, idxo], simi)

def RankRightFn(tparams, options):
    idxl=tensor.scalar('idxl', dtype='int64')
    idxo=tensor.scalar('idxo',dtype='int64')

    embL = tparams['Wemb'][idxl].reshape([1, 1, options['dim_proj']])
    embO = tparams['Wemb'][idxo].reshape([1, 1, options['dim_proj']])
    emb=tensor.concatenate([embL, embO])
    emb=tensor.tile(emb, [1, options['n_ent'], 1])
    out = tparams['Wemb'][tensor.arange(options['n_ent'])].reshape([options['n_ent'], options['dim_proj']])

    time_steps = emb.shape[0]
    n_samples = emb.shape[1]
    mask = tensor.alloc(numpy_floatX(1), time_steps, n_samples)
    proj = get_layer(options['encoder'])(tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    proj = proj[-1,:,:]
    simi= L2sim(proj, out)
        
    return theano.function([idxl, idxo], simi)

def train_lstm(
    dim_proj=20,  # word embeding dimension
    max_epochs=201,  # The maximum number of epoch to run
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    margin=0.5,
    lambda_pen = 0,
    penalty_norm=0,
    alpha=0,
    optimizer=sgd,  # sgd, adadelta and rmsprop
    encoder='rtranse',
    validFreq=10,  # Compute the validation error after this number of epochs.
    dataset='SN',
    datapath='datapath/', 
    savepath= '',
    # Parameter for extra option
    reload_model=''
):

    numpy.random.seed(1234)
    random.seed(1234)

    Nsyn=728
    Nrel=7
    Nent=Nsyn-Nrel
    batch_size=200
        
    # Model options
    model_options = locals().copy()
    model_options['n_words'] = Nsyn
    model_options['n_ent'] = Nent
    print "model options", model_options

    print 'Loading data'
    trainl = convert2idx(load_file(datapath + dataset + '-train-lhs.pkl'))
    trainr = convert2idx(load_file(datapath + dataset + '-train-rhs.pkl'))
    traino = convert2idx(load_file(datapath + dataset + '-train-rel.pkl'))
    train_lex, labelsTrain = buildTriplesForward(trainl,trainr,traino)

    validl = convert2idx(load_file(datapath + dataset + '-valid-lhs.pkl'))
    validr = convert2idx(load_file(datapath + dataset + '-valid-rhs.pkl'))
    valido = convert2idx(load_file(datapath + dataset + '-valid-rel.pkl'))

    testl = convert2idx(load_file(datapath + dataset + '-test-lhs.pkl'))
    testr = convert2idx(load_file(datapath + dataset + '-test-rhs.pkl'))
    o = convert2idx(load_file(datapath + dataset + '-test-rel.pkl')[-Nrel:, :])
    testo = convert2idx(load_file(datapath + dataset + '-test-rel.pkl'))

    alt_paths=cPickle.load(open(datapath+'alt_paths.pkl'))
    altrel2idx=cPickle.load(open(datapath+'altrel2idx.pkl'))
    alphas=numpy.asarray(cPickle.load(open(datapath+'alphas.pkl')))
    
    true_triples=numpy.concatenate([testl,validl,trainl,testo,valido,traino,testr,validr,trainr]).reshape(3,testl.shape[0]+validl.shape[0]+trainl.shape[0]).T   

    print 'Building model'
    params = init_params(model_options)
    tparams = init_tparams(params)

    (xP, xN, mask, yP, yN, cost) = build_model(tparams, model_options)
    (x, y, maskAlt, weight, costTotal) = autoencoding_errors(tparams, model_options, cost)

    f_cost = theano.function([xP, xN, mask, yP, yN, x, y, weight, maskAlt], costTotal, name='f_cost')
    
    grads = tensor.grad(costTotal, wrt=tparams.values())
    f_grad = theano.function([xP, xN, mask, yP, yN, x, y, weight, maskAlt], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        xP, xN, mask, yP, yN, x, y, maskAlt, costTotal, weight)
    ranklfunc=RankLeftFn(tparams, model_options)
    rankrfunc=RankRightFn(tparams, model_options)
    
    print 'Optimization'
    best_MR=numpy.inf
    try:
        for eidx in xrange(max_epochs):
            print "Epoch %s"%(eidx)
            trainln = create_random_arr(len(train_lex), Nent)
            trainrn = create_random_arr(len(train_lex), Nent)

            train_lex_neg, labelsTrain_neg = buildTriplesNeg(trainln,trainrn,traino)

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train_lex), batch_size, shuffle=True)
            for _, train_index in kf:

                # Select the random examples for this minibatch
                x = [train_lex[t]for t in train_index]
                y = [labelsTrain[t]for t in train_index]
                xP, mask, yP = prepare_data(x, y)
                                
                x = [train_lex_neg[t]for t in train_index]
                y = [labelsTrain_neg[t]for t in train_index]
                xN, mask, yN = prepare_data(x, y)

                x, mask2hop, list_idx=build_matrices(alt_paths, xP[0,:], xP[1,:], numpy.asarray(yP), altrel2idx)
                x2hop = x[:-1,:]
                y2hop = list(x[-1,:])
                
                costTT = f_grad_shared(xP, xN, mask, yP, yN, x2hop, y2hop, mask2hop, alphas[list_idx])
                f_update(lrate)

            if numpy.mod(eidx, validFreq) == 0:
                #VALIDATION PERFORMANCE
                resvalid = FilteredRankingScoreIdx(ranklfunc, rankrfunc, validl, validr, valido, true_triples)
                MR = numpy.mean(resvalid[0]+resvalid[1])
                
                if MR < best_MR:
                    best_MR=MR
                    #TEST PERFORMANCE
                    restest = FilteredRankingScoreIdx(ranklfunc, rankrfunc, testl, testr, testo, true_triples)
                    test_MR = numpy.mean(restest[0]+restest[1])
                    test_HITS5 = numpy.mean(numpy.asarray(restest[0] + restest[1]) <= 5) * 100
                    #saveto=''
                    #params = unzip(tparams)
                    #numpy.savez(saveto,  **params)

    except KeyboardInterrupt:
        print "Training interupted"

    print "TEST MR: %s"%(test_MR)
    print "TEST HITS@5: %s"%(test_HITS5)

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm()
