#! /usr/bin/python
import sys
from model import *

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


def RankingEval(datapath='../data/prescription/', dataset='FB15k-test',
        loadmodel='best_valid_model.pkl', neval='all', Nsyn=14951, n=10,
        idx2synsetfile='FB15k_idx2entity.pkl'):

    # Load model
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    f.close()

    # Load data
    l = load_file(datapath + dataset + '-lhs.pkl')
    r = load_file(datapath + dataset + '-rhs.pkl')
    o = load_file(datapath + dataset + '-rel.pkl')

    trainl = load_file(datapath + dataset + '-train-lhs.pkl')
    trainr = load_file(datapath + dataset + '-train-rhs.pkl')
    traino = load_file(datapath + dataset + '-train-rel.pkl')
    validl = load_file(datapath + dataset + '-valid-lhs.pkl')
    validr = load_file(datapath + dataset + '-valid-rhs.pkl')
    valido = load_file(datapath + dataset + '-valid-rel.pkl')
    testl = load_file(datapath + dataset + '-test-lhs.pkl')
    testr = load_file(datapath + dataset + '-test-rhs.pkl')
    testo = load_file(datapath + dataset + '-test-rel.pkl')


    if type(embeddings) is list:
        traino = traino[-embeddings[1].N:, :]
        valido = valido[-embeddings[1].N:, :]
        testo = testo[-embeddings[1].N:, :]

    # Convert sparse matrix to indexes
    if neval == 'all':
        idxl = convert2idx(l)
        idxr = convert2idx(r)
        idxo = convert2idx(o)
    else:
        idxl = convert2idx(l)[:neval]
        idxr = convert2idx(r)[:neval]
        idxo = convert2idx(o)[:neval]

    true_triples = 

    rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=Nsyn)

    # FilteredRightPredictionIdx(sl, sr, idxl, idxo, idxr, true_triples): TODO complete the function call
    res = Filtered(rankrfunc, idxl, idxr, idxo, true_triples)

