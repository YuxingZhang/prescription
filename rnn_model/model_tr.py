# use linear 1-hidden layer neural network to combine lhs embedding with RNN embedding, then output the lhs embedding
import numpy as np
import theano
import theano.tensor as T
import lasagne

from collections import OrderedDict
from settings import CHAR_DIM, C2W_HDIM, WDIM, SCALE, N_BATCH, GRAD_CLIP, REGULARIZATION, LEARNING_RATE, MOMENTUM, GAMMA

NL1 = lasagne.nonlinearities.sigmoid
NL2 = lasagne.nonlinearities.tanh
NL3 = lasagne.nonlinearities.tanh
LR = lasagne.regularization.l2
WDIM = WDIM / 2

# margin cost defined in TransE
def margincost(pos_loss, neg_loss, margin):
    out = margin + pos_loss - neg_loss
    return T.sum(out * (out > 0))

# L2 distance between two Theano tensors, compute L2 distance for every row
def L2dist(left, right):
    return T.sqrt(T.sum(T.sqr(left - right), axis=1))

class charLM(object):
    def __init__(self, n_char, n_lhs, n_rel, n_rhs, emb_dim=WDIM, pretrained=None): # is WDIM the RNN embedding dimension? yes
        # params
        if pretrained==None:
            self.params = OrderedDict()
            self.params = init_params(self.params, n_char, n_lhs, n_rel, n_rhs, emb_dim) # define n_rhs, emb_dim
        else:
            self.params = load_params_shared(pretrained)

        self.n_rhs = n_rhs

        # model
        in_lhs, in_lhsn, emb_lhs, emb_lhsn = embedding_lhs(self.params, n_lhs, emb_dim)
        in_rhs, in_rhsn, emb_rhs, emb_rhsn = embedding_rhs(self.params, n_rhs, emb_dim)
        in_rel, emb_rel = embedding_rel(self.params, n_rel, emb_dim)
        
        # N_BATCH for input size? or just None, because later we need to do validation and testing, can uses any size
        # up to this point, we have emb_lhs, emb_lhsn, emb_rhs, emb_rhsn, emb_rel
        # define loss
        pred_rhs = emb_lhs + emb_rel # true lhs + rel
        pred_lhs = emb_lhsn + emb_rel # negative lhs + rel
        pred_rel = emb_rhs - emb_lhs  # predicted relation, rhs - lhs, for visualization

        # TODO remove the dist(lhs, rhs - rel) terms in the loss function
        pos_loss_r = L2dist(pred_rhs, emb_rhs) # positive triple distance
        pos_loss_l = L2dist(emb_lhs, emb_rhs - emb_rel) # TODO remove
        neg_loss_r = L2dist(pred_rhs, emb_rhsn) # negative triple distance with corrupted rhs
        #neg_loss_l = L2dist(pred_lhs, emb_rhs) # negative triple distance with corrupted lhs TODO uncomment
        neg_loss_l = L2dist(emb_lhsn, emb_rhs - emb_rel) # negative triple distance with corrupted lhs
        loss_rn = margincost(pos_loss_r, neg_loss_r, GAMMA) # GAMMA is the margin, GAMMA = 1.0 in TransE
        loss_ln = margincost(pos_loss_l, neg_loss_l, GAMMA) # TODO replace pos_loss_l with pos_loss_r
        loss = loss_rn + loss_ln
        # do we need loss_ln? Yes, and how do we sample random lhs embedding? build a dict too
        self.cost = T.mean(loss)
        # can we only add regularization to the RNN parameters? yes, only pass RNN parameters
        cost_only = T.mean(loss)

        '''get_output can specify input, so don't need to define another embedding layer'''

        # updates
        self.lr = LEARNING_RATE
        self.mu = MOMENTUM
        updates = lasagne.updates.nesterov_momentum(self.cost, self.params.values(), self.lr, momentum=self.mu)
        # try different lr, momentum

        # theano functions
        self.inps = [in_lhs, in_lhsn, in_rel, in_rhs, in_rhsn] # inputs for the function
        self.cost_fn = theano.function(self.inps,cost_only)
        self.encode_fn = theano.function([in_lhs], emb_lhs) # compute RNN embeddings given word (drug name)
        self.train_fn = theano.function(self.inps,self.cost,updates=updates)
        self.pred_right_fn = theano.function([in_lhs, in_rel], pred_rhs) # compute lhs + rel as predicted rhs
        self.emb_right_fn = theano.function([in_rhs], emb_rhs) # compute only rhs embedding
        self.pred_rel_fn = theano.function([in_lhs, in_rhs], pred_rel)

    def pred_rel(self, in_lhs, in_rhs):
        return self.pred_rel_fn(in_lhs, in_rhs)

    def train(self, in_lhs, in_lhsn, in_rel, in_rhs, in_rhsn):
        return self.train_fn(in_lhs, in_lhsn, in_rel, in_rhs, in_rhsn)

    def validate(self, in_lhs, in_lhsn, in_rel, in_rhs, in_rhsn):
        return self.cost_fn(in_lhs, in_lhsn, in_rel, in_rhs, in_rhsn)

    def compute_emb_right_all(self): # compute a (n_rhs * emb_dim) numpy matrix, each row is an embedding for a right hand side entity
        in_rhs_all = np.arange(self.n_rhs).astype('int32') # input pretend to compute the embedding for all right hand side entities
        self.emb_right_all = self.emb_right_fn(in_rhs_all)

    def encode(self, in_lhs):
        return self.encode_fn(in_lhs)

    def rank_right(self, in_lhs, in_rel, in_rhs): # return a len(in_lhs) size list, each element is the rank of the true rhs among all the rhs
        pred_rhs_batch = self.pred_right_fn(in_lhs, in_rel)
        right_ranks = []
        for i in range(pred_rhs_batch.shape[0]):
            true_idx = in_rhs[i]
            distances = np.zeros(self.emb_right_all.shape[0])
            for j in range(self.emb_right_all.shape[0]):
                distances[j] = np.linalg.norm(pred_rhs_batch[i, :] - self.emb_right_all[j, :], 2)
            rank = np.argsort(np.argsort(distances))
            right_ranks += [rank[true_idx]]
        return right_ranks

    def update_learningrate(self):
        self.lr = max(1e-5,self.lr / 2)
        updates = lasagne.updates.nesterov_momentum(self.cost, self.params.values(), self.lr, momentum=self.mu)
        self.train_fn = theano.function(self.inps,self.cost,updates=updates)

    def save_model(self,save_path):
        saveparams = OrderedDict()
        for kk,vv in self.params.iteritems():
            saveparams[kk] = vv.get_value()
        np.savez(save_path,**saveparams)

    def print_params(self):
        for kk,vv in self.params.iteritems():
            print("Param {} Max {} Min {}".format(kk, np.max(vv.get_value()), np.min(vv.get_value())))

def init_params(params, n_char, n_lhs, n_rel, n_rhs, emb_dim):
    np.random.seed(0)

    # lookup table # TODO when using float 32, there will be an error in theano 
    # "An update must have the same type as the original shared variable", why is that

    # Initialize parameters for lhs entity embedding
    params['W_emb_lhs'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(n_lhs, emb_dim)).astype('float64'), name='W_emb_lhs')

    # Initialize parameters for rhs entity embedding
    params['W_emb_rhs'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(n_rhs, emb_dim)).astype('float64'), name='W_emb_rhs')

    # Initialize parameters for relation embedding
    params['W_emb_rel'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(n_rel, emb_dim)).astype('float64'), name='W_emb_rel')

    # Initialize parameters for dense layer
    return params

# by Yuxing Zhang
def embedding_rhs(params, n_rhs, emb_dim):
    '''
    Embedding part for right hand side entity embedding and right hand side negative entity embedding

    :param params: dict to store parameters
    '''
    # input variables that is right hand side entity
    emb_in_rhs = T.ivector() # B * 1 vector, where each row is a number between 0 and (n_rhs - 1) as the index
    emb_in_rhsn = T.ivector() # B * 1 vector, where each row is a number between 0 and (n_rhs - 1) as the index

    # Input layer over entity
    l_in_rhs = lasagne.layers.InputLayer(shape=(N_BATCH, ), name = 'rhs_input') # removing input_var to reuse it for negative rhs

    # Embedding layer for rhs entity, and emb_dim should equal # the embedding dimension from RNN model.
    l_emb_rhs = lasagne.layers.EmbeddingLayer(l_in_rhs, input_size=n_rhs, output_size=emb_dim, W=params['W_emb_rhs'])

    return emb_in_rhs, emb_in_rhsn, lasagne.layers.get_output(l_emb_rhs, emb_in_rhs), lasagne.layers.get_output(l_emb_rhs, emb_in_rhsn)

# by Yuxing Zhang
def embedding_rel(params, n_rel, emb_dim):
    '''
    Embedding part for right hand side entity embedding

    :param params: dict to store parameters
    '''
    # input variables that is the relation index
    emb_in_rel = T.ivector() # B * 1 vector, where each row is a number between 0 and (n_rel - 1) as the index

    # Input layer over relation
    l_in_rel = lasagne.layers.InputLayer(shape=(N_BATCH, ), input_var=emb_in_rel, name = 'rel_input')

    # Embedding layer for relation, and emb_dim should equal # the embedding dimension from RNN model.
    l_emb_rel = lasagne.layers.EmbeddingLayer(l_in_rel, input_size=n_rel, output_size=emb_dim, W=params['W_emb_rel'])

    return emb_in_rel, lasagne.layers.get_output(l_emb_rel)

# by Yuxing Zhang
def embedding_lhs(params, n_lhs, emb_dim):
    '''
    Embedding part for left hand side entity embedding and left hand side negative entity embedding

    :param params: dict to store parameters
    '''
    # input variables that is right hand side entity
    emb_in_lhs = T.ivector() # B * 1 vector, where each row is a number between 0 and (n_rhs - 1) as the index
    emb_in_lhsn = T.ivector() # B * 1 vector, where each row is a number between 0 and (n_rhs - 1) as the index

    # Input layer over entity
    l_in_lhs = lasagne.layers.InputLayer(shape=(N_BATCH, ), name = 'lhs_input') # removing input_var to reuse it for negative rhs

    # Embedding layer for rhs entity, and emb_dim should equal # the embedding dimension from RNN model.
    l_emb_lhs = lasagne.layers.EmbeddingLayer(l_in_lhs, input_size=n_lhs, output_size=emb_dim, W=params['W_emb_lhs'])
    # extra input for unseen entities 0

    return emb_in_lhs, emb_in_lhsn, lasagne.layers.get_output(l_emb_lhs, emb_in_lhs), lasagne.layers.get_output(l_emb_lhs, emb_in_lhsn)

def load_params(path):
    """
    Load previously saved model
    """
    params = OrderedDict()

    with open(path,'r') as f:
        npzfile = np.load(f)
        for kk, vv in npzfile.iteritems():
            params[kk] = vv

    return params

def load_params_shared(path):
    """
    Load previously saved model
    """
    params = OrderedDict()

    with open(path,'r') as f:
        npzfile = np.load(f)
        for kk, vv in npzfile.iteritems():
            params[kk] = theano.shared(vv, name=kk)

    return params
