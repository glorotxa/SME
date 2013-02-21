#! /usr/bin/python

from model import *


# Utils ----------------------------------------------------------------------
def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)


def compute_prauc(pred, lab):
    pred = np.asarray(pred)
    lab = np.asarray(lab)

    order = np.argsort(pred)
    lab_ordered = lab[order]
    pred_ordered = pred[order]

    precision = {}
    recall = {}
    # All examples are classified 1
    precision[np.min(pred_ordered) - 1.0] = (np.sum(lab_ordered) /
            float(len(lab)))
    recall[np.min(pred_ordered) - 1.0] = 1.
    for i in range(len(lab)):
        if len(lab) - i - 1 == 0:
            # No examples are classified 1
            precision[pred_ordered[i]] = 1
        else:
            precision[pred_ordered[i]] = (np.sum(lab_ordered[i + 1:]) /
                    float(len(lab) - i - 1))
        recall[pred_ordered[i]] = (np.sum(lab_ordered[i + 1:]) /
                float(np.sum(lab_ordered)))

    # Precision-Recall curve points
    points = []
    for i in np.sort(precision.keys())[::-1]:
        points += [(float(recall[i]), float(precision[i]))]
    # Compute area
    auc = sum((y0 + y1) / 2. * (x1 - x0) for (x0, y0), (x1, y1) in
            zip(points[:-1], points[1:]))
    return auc


class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z
# ----------------------------------------------------------------------------


# Experiment function --------------------------------------------------------
def Tensorexp(state, channel):

    # Show experiment parameters
    print >> sys.stderr, state
    np.random.seed(state.seed)

     # Experiment folder
    if hasattr(channel, 'remote_path'):
        state.savepath = channel.remote_path + '/'
    elif hasattr(channel, 'path'):
        state.savepath = channel.path + '/'
    else:
        if not os.path.isdir(state.savepath):
            os.mkdir(state.savepath)

     # Positives
    trainl = load_file(state.datapath + state.dataset +
            '-train-pos-lhs-fold%s.pkl' % state.fold)
    trainr = load_file(state.datapath + state.dataset +
            '-train-pos-rhs-fold%s.pkl' % state.fold)
    traino = load_file(state.datapath + state.dataset +
            '-train-pos-rel-fold%s.pkl' % state.fold)
    if state.op == 'SE':
        traino = traino[-state.Nrel:, :]

    # Negatives
    trainln = load_file(state.datapath + state.dataset +
            '-train-neg-lhs-fold%s.pkl' % state.fold)
    trainrn = load_file(state.datapath + state.dataset +
            '-train-neg-rhs-fold%s.pkl' % state.fold)
    trainon = load_file(state.datapath + state.dataset +
            '-train-neg-rel-fold%s.pkl' % state.fold)
    if state.op == 'SE':
        trainon = trainon[-state.Nrel:, :]

    # Valid set
    validl = load_file(state.datapath + state.dataset +
            '-valid-lhs-fold%s.pkl' % state.fold)
    validr = load_file(state.datapath + state.dataset +
            '-valid-rhs-fold%s.pkl' % state.fold)
    valido = load_file(state.datapath + state.dataset +
            '-valid-rel-fold%s.pkl' % state.fold)
    if state.op == 'SE':
        valido = valido[-state.Nrel:, :]
    outvalid = cPickle.load(open(state.datapath +
        '%s-valid-targets-fold%s.pkl' % (state.dataset, state.fold)))

    # Test set
    testl = load_file(state.datapath + state.dataset +
            '-test-lhs-fold%s.pkl' % state.fold)
    testr = load_file(state.datapath + state.dataset +
            '-test-rhs-fold%s.pkl' % state.fold)
    testo = load_file(state.datapath + state.dataset +
            '-test-rel-fold%s.pkl' % state.fold)
    if state.op == 'SE':
        testo = testo[-state.Nrel:, :]
    outtest = cPickle.load(open(state.datapath +
        '%s-test-targets-fold%s.pkl' % (state.dataset, state.fold)))

    # Model declaration
    if not state.loadmodel:
        # operators
        if state.op == 'Unstructured':
            leftop = Unstructured()
            rightop = Unstructured()
        elif state.op == 'SME_lin':
            leftop = LayerLinear(np.random, 'lin', state.ndim, state.ndim,
                    state.nhid, 'left')
            rightop = LayerLinear(np.random, 'lin', state.ndim, state.ndim,
                    state.nhid, 'right')
        elif state.op == 'SME_bil':
            leftop = LayerBilinear(np.random, 'lin', state.ndim, state.ndim,
                    state.nhid, 'left')
            rightop = LayerBilinear(np.random, 'lin', state.ndim, state.ndim,
                    state.nhid, 'right')
        elif state.op == 'SE':
            leftop = LayerMat('lin', state.ndim, state.nhid)
            rightop = LayerMat('lin', state.ndim, state.nhid)
        # embeddings
        if not state.loademb:
            embeddings = Embeddings(np.random, state.Nent, state.ndim, 'emb')
        else:
            f = open(state.loademb)
            embeddings = cPickle.load(f)
            f.close()
        if state.op == 'SE' and type(embeddings) is not list:
            relationl = Embeddings(np.random, state.Nrel,
                    state.ndim * state.nhid, 'rell')
            relationr = Embeddings(np.random, state.Nrel,
                    state.ndim * state.nhid, 'relr')
            embeddings = [embeddings, relationl, relationr]
        simfn = eval(state.simfn + 'sim')
    else:
        f = open(state.loadmodel)
        embeddings = cPickle.load(f)
        leftop = cPickle.load(f)
        rightop = cPickle.load(f)
        simfn = cPickle.load(f)
        f.close()

    # Functions compilation
    trainfunc = TrainFn(simfn, embeddings, leftop, rightop, marge=state.marge)
    testfunc = SimFn(simfn, embeddings, leftop, rightop)

    out = []
    outb = []
    state.bestvalid = -1

    batchsize = trainl.shape[1] / state.nbatches

    print >> sys.stderr, "BEGIN TRAINING"
    timeref = time.time()
    for epoch_count in xrange(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(trainl.shape[1])
        trainl = trainl[:, order]
        trainr = trainr[:, order]
        traino = traino[:, order]
        order = np.random.permutation(trainln.shape[1])
        trainln = trainln[:, order]
        trainrn = trainrn[:, order]
        trainon = trainon[:, order]

        for i in range(state.nbatches):
            tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
            tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
            tmpo = traino[:, i * batchsize:(i + 1) * batchsize]
            tmpln = trainln[:, i * batchsize:(i + 1) * batchsize]
            tmprn = trainrn[:, i * batchsize:(i + 1) * batchsize]
            tmpon = trainon[:, i * batchsize:(i + 1) * batchsize]
            # training iteration
            outtmp = trainfunc(state.lremb, state.lrparam / float(batchsize),
                    tmpl, tmpr, tmpo, tmpln, tmprn, tmpon)
            out += [outtmp[0] / float(batchsize)]
            outb += [outtmp[1]]
            # embeddings normalization
            if type(embeddings) is list:
                embeddings[0].normalize()
            else:
                embeddings.normalize()

        if (epoch_count % state.test_all) == 0:
            # model evaluation
            print >> sys.stderr, "-- EPOCH %s (%s seconds per epoch):" % (
                    epoch_count,
                    round(time.time() - timeref, 3) / float(state.test_all))
            timeref = time.time()
            print >> sys.stderr, "COST >> %s +/- %s, %% updates: %s%%" % (
                    round(np.mean(out), 4), round(np.std(out), 4),
                    round(np.mean(outb) * 100, 3))
            out = []
            outb = []
            valsim = testfunc(validl, validr, valido)[0]
            state.valid = compute_prauc(valsim, outvalid)
            print >> sys.stderr, "\tPR AUC >> valid: %s" % (state.valid)
            if state.bestvalid == -1 or state.valid > state.bestvalid:
                testsim = testfunc(testl, testr, testo)[0]
                state.besttest = compute_prauc(testsim, outtest)
                state.bestvalid = state.valid
                state.bestepoch = epoch_count
                # Save model best valid model
                f = open(state.savepath + '/best_valid_model.pkl', 'w')
                cPickle.dump(embeddings, f, -1)
                cPickle.dump(leftop, f, -1)
                cPickle.dump(rightop, f, -1)
                cPickle.dump(simfn, f, -1)
                f.close()
                print >> sys.stderr, "\t\t##### NEW BEST VALID >> test: %s" % (
                        state.besttest)
            # Save current model
            f = open(state.savepath + '/current_model.pkl', 'w')
            cPickle.dump(embeddings, f, -1)
            cPickle.dump(leftop, f, -1)
            cPickle.dump(rightop, f, -1)
            cPickle.dump(simfn, f, -1)
            f.close()
            state.nbepochs = epoch_count
            print >> sys.stderr, "\t(the evaluation took %s seconds)" % (
                round(time.time() - timeref, 3))
            timeref = time.time()
            channel.save()
    return channel.COMPLETE


def launch(datapath='data/', dataset='umls', fold=0, Nent=184,
        Nrel=49, loadmodel=False, loademb=False, op='Unstructured',
        simfn='Dot', ndim=50, nhid=50, marge=1., lremb=0.1, lrparam=1.,
        nbatches=100, totepochs=2000, test_all=1, seed=666, savepath='.'):

    # Argument of the experiment script
    state = DD()

    state.datapath = datapath
    state.dataset = dataset
    state.fold = fold
    state.Nent = Nent
    state.Nrel = Nrel
    state.loadmodel = loadmodel
    state.loademb = loademb
    state.op = op
    state.simfn = simfn
    state.ndim = ndim
    state.nhid = nhid
    state.marge = marge
    state.lremb = lremb
    state.lrparam = lrparam
    state.nbatches = nbatches
    state.totepochs = totepochs
    state.test_all = test_all
    state.seed = seed
    state.savepath = savepath

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    # Jobman channel remplacement
    class Channel(object):
        def __init__(self, state):
            self.state = state
            f = open(self.state.savepath + '/orig_state.pkl', 'w')
            cPickle.dump(self.state, f, -1)
            f.close()
            self.COMPLETE = 1

        def save(self):
            f = open(self.state.savepath + '/current_state.pkl', 'w')
            cPickle.dump(self.state, f, -1)
            f.close()

    channel = Channel(state)

    Tensorexp(state, channel)

if __name__ == '__main__':
    launch()
