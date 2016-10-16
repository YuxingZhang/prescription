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

# Layers ----------------------------------------------------------------------

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
        lhs = (embedding.E[:, :subtensorspec]).T # only consider entities not relations
    else:
        lhs = embedding.E.T
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    #print >> sys.stderr, "rell size = ", rell.shape
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    tmp = rightop(rhs, relr) # in the case of TransE, this part is jus rhs
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


def TrainFn1Member(fnsim, embeddings, leftop, rightop, marge=1.0, rel=False):
    """
    This function returns a theano function to perform a training iteration,
    contrasting positive and negative triplets. members are given as sparse
    matrices. For one positive triplet there are two or three (if rel == True)
    negative triplets. To create a negative triplet we replace only one member
    at a time.

    :param fnsim: similarity function (on theano variables). See line 20 for L2sim function
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

    if hasattr(fnsim, 'params'): # always false
        # If the similarity function has some parameters, we update them too.
        gradientsparams = T.grad(cost,
            leftop.params + rightop.params + fnsim.params)
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params + fnsim.params, gradientsparams))
    else:
        gradientsparams = T.grad(cost, leftop.params + rightop.params) # nothing happens here, since neither op has parameters
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params, gradientsparams))

    gradients_embedding = T.grad(cost, embedding.E) # TODO add rnn parameters
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
                 triplet 'left' member, shape=(#examples,N [Embeddings]). # TODO replace with RNN input
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
        errl += [np.argsort(np.argsort((sl(r, o)[0]).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]
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
    #print >> sys.stderr, "-----------> printing true_triples!"
    #print >> sys.stderr, true_triples[100, :]
    for l, o, r in zip(idxl, idxo, idxr):
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,) # a list of positions k where left[k] == l
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,) # a list of positions k where rel[k] == o
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,) # a list of positions k where right[k] == r
 
        inter_l = [i for i in ir if i in io]
        # corrupted triplets that actually valid, except the current one with left = l (since we need this rank), 
        # and this set is the set of triplets that appear somewhere in the dataset
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
    
def FilteredRightPredictionIdx(sr, idxl, idxo, idxr, true_triples):
    """
    This function computes the predictions of rhs by returning the entity with the highest
    score, over a list of lhs, rhs and rel indexes.

    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices, which are the true rhs, include this because we are 
    filtering the entities that appear in the training set, validation set or the test set and
    we don't want to remove the true entity from the list.
    :param idxo: list of relation indices.
    """

    predictions = []
    #print >> sys.stderr, "-----------> printing true_triples!"
    #print >> sys.stderr, true_triples[100, :]
    for l, o, r in zip(idxl, idxo, idxr):
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,) # a list of positions k where left[k] == l
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,) # a list of positions k where rel[k] == o
 
        inter_r = [i for i in il if i in io]
        # corrupted triplets that actually valid, except the current one with right = r (since we will), 
        # and this set is the set of triplets that appear somewhere in the dataset
        rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]
        scores_r = (sr(l, o)[0]).flatten()
        scores_r[rmv_idx_r] = -np.inf
        predictions += [np.argsort(-scores_r).flatten()[0]] # the index with the highest score
    return predictions
    
# ----------------------------------------------------------------------------
