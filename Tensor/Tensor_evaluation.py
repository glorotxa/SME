#! /usr/bin/python

from model import *


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


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


def PRAUCEval(datapath='../data/', dataset='umls-test',
        loadmodel='best_valid_model.pkl', fold=0):

    # Load model
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    f.close()

    # Load data
    l = load_file(datapath + dataset + '-lhs-fold%s.pkl' % fold)
    r = load_file(datapath + dataset + '-rhs-fold%s.pkl' % fold)
    o = load_file(datapath + dataset + '-rel-fold%s.pkl' % fold)
    if type(embeddings) is list:
        o = o[-embeddings[1].N:, :]
    out = cPickle.load(open(datapath + '%s-targets-fold%s.pkl' %
        (dataset, fold)))

    func = SimFn(simfn, embeddings, leftop, rightop)
    sim = func(l, r, o)[0]

    AUC = compute_prauc(list(sim), list(out))
    print "### Prediction Recall AUC:", AUC

    return AUC


if __name__ == '__main__':
    PRAUCEval()
