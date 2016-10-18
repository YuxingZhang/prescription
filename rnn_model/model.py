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

# margin cost defined in TransE
def margincost(pos_loss, neg_loss, margin):
    out = margin + pos_loss - neg_loss
    return T.sum(out * (out > 0))

# L2 distance between two Theano tensors, compute L2 distance for every row
def L2dist(left, right):
    return T.sqrt(T.sum(T.sqr(left - right), axis=1))

class charLM(object):
    def __init__(self, n_char, n_voc, n_rel, emb_dim=WDIM, pretrained=None): # is WDIM the RNN embedding dimension? yes
        # params
        if pretrained==None:
            self.params = OrderedDict()
            self.params = init_params(self.params, n_char, n_voc, n_rel, emb_dim) # define n_voc, emb_dim
        else:
            self.params = load_params_shared(pretrained)

        self.n_voc = n_voc

        # model
        in_lhs, in_lmask, in_lhsn, in_lmaskn, emb_lhs, emb_lhsn, l_encoder = char2vec(self.params, n_char) 
        # TODO maybe concatenate RNN embedding with look up table? Do it later.
        in_rhs, in_rhsn, emb_rhs, emb_rhsn = embedding_rhs(self.params, n_voc, emb_dim)
        in_rel, emb_rel = embedding_rel(self.params, n_rel, emb_dim)
        
        # TODO N_BATCH for input size? or just None, because later we need to do validation and testing
        # define loss
        pred_rhs = emb_lhs + emb_rel
        pos_loss = L2dist(pred_rhs, emb_rhs) # positive triple distance
        neg_loss_r = L2dist(pred_rhs, emb_rhsn) # negative triple distance
        loss_rn = margincost(pos_loss, neg_loss_r, GAMMA) # GAMMA is the margin
        loss = loss_rn
        # TODO do we need loss_ln? And how do we sample random lhs embedding? yes build a dict too
        self.cost = T.mean(loss) + REGULARIZATION*lasagne.regularization.apply_penalty(lasagne.layers.get_all_params(l_encoder), LR)
        # can we only add regularization to the RNN parameters? yes, only pass RNN parameters
        cost_only = T.mean(loss)

        '''get_output can specify input, so don't need to define another embedding layer'''

        # updates
        self.lr = LEARNING_RATE
        self.mu = MOMENTUM
        updates = lasagne.updates.nesterov_momentum(self.cost, self.params.values(), self.lr, momentum=self.mu)

        # theano functions
        self.inps = [in_lhs, in_lmask, in_rel, in_rhs, in_rhsn] # inputs for the function
        # TODO add loss_ln, self.inps = [in_lhs, in_lmask, in_lhsn, in_lmaskn, in_rel, in_rhs, in_rhsn]
        self.cost_fn = theano.function(self.inps,cost_only)
        self.encode_fn = theano.function([in_lhs, in_lmask], emb_lhs) # compute RNN embeddings given word (drug name)
        self.train_fn = theano.function(self.inps,self.cost,updates=updates)
        self.pred_right_fn = theano.function([in_lhs, in_lmask, in_rel], pred_rhs) # compute lhs + rel as predicted rhs
        self.emb_right_fn = theano.function([in_rhs], emb_rhs) # compute only rhs embedding

    def train(self, in_lhs, in_lmask, in_rel, in_rhs, in_rhsn):
        return self.train_fn(in_lhs, in_lmask, in_rel, in_rhs, in_rhsn)

    def validate(self, in_lhs, in_lmask, in_rel, in_rhs, in_rhsn):
        return self.cost_fn(in_lhs, in_lmask, in_rel, in_rhs, in_rhsn)

    def compute_emb_right_all(self): # compute a (n_voc * emb_dim) numpy matrix, each row is an embedding for a right hand side entity
        in_rhs_all = np.arange(self.n_voc).astype('int32') # input pretend to compute the embedding for all right hand side entities
        self.emb_right_all = self.emb_right_fn(in_rhs_all)

    def encode(self, w, m):
        return self.encode_fn(w,m)

    def rank_right(self, in_lhs, in_lmask, in_rel, in_rhs): # return a len(in_lhs) size list, each element is the rank of the true rhs
        pred_rhs_batch = self.pred_right_fn(in_lhs, in_lmask, in_rel)
        print pred_rhs_batch.shape
        print pred_rhs_batch
        print self.emb_right_all.shape
        print self.emb_right_all
        right_ranks = []
        

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

def init_params(params, n_char, n_voc, n_rel, emb_dim):
    np.random.seed(0)

    # lookup table # TODO when using float 32, there will be an error in theano 
    # "An update must have the same type as the original shared variable", why is that
    params['Wc'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(n_char,CHAR_DIM)).astype('float64'), name='Wc')

    # f-GRU
    params['W_c2w_f_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float64'), name='W_c2w_f_r')
    params['W_c2w_f_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float64'), name='W_c2w_f_z')
    params['W_c2w_f_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float64'), name='W_c2w_f_h')
    params['b_c2w_f_r'] = theano.shared(np.zeros((C2W_HDIM)).astype('float64'), name='b_c2w_f_r')
    params['b_c2w_f_z'] = theano.shared(np.zeros((C2W_HDIM)).astype('float64'), name='b_c2w_f_z')
    params['b_c2w_f_h'] = theano.shared(np.zeros((C2W_HDIM)).astype('float64'), name='b_c2w_f_h')
    params['U_c2w_f_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float64'), name='U_c2w_f_r')
    params['U_c2w_f_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float64'), name='U_c2w_f_z')
    params['U_c2w_f_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float64'), name='U_c2w_f_h')

    # b-GRU
    params['W_c2w_b_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float64'), name='W_c2w_b_r')
    params['W_c2w_b_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float64'), name='W_c2w_b_z')
    params['W_c2w_b_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float64'), name='W_c2w_b_h')
    params['b_c2w_b_r'] = theano.shared(np.zeros((C2W_HDIM)).astype('float64'), name='b_c2w_b_r')
    params['b_c2w_b_z'] = theano.shared(np.zeros((C2W_HDIM)).astype('float64'), name='b_c2w_b_z')
    params['b_c2w_b_h'] = theano.shared(np.zeros((C2W_HDIM)).astype('float64'), name='b_c2w_b_h')
    params['U_c2w_b_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float64'), name='U_c2w_b_r')
    params['U_c2w_b_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float64'), name='U_c2w_b_z')
    params['U_c2w_b_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float64'), name='U_c2w_b_h')

    # dense
    params['W_c2w'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(2*C2W_HDIM,WDIM)).astype('float64'), name='W_c2w_df')
    params['b_c2w'] = theano.shared(np.zeros((WDIM)).astype('float64'), name='b_c2w_df')

    # Initialize parameters for rhs entity embedding
    params['W_emb_rhs'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(n_voc, emb_dim)).astype('float64'), name='W_emb_rhs')

    # Initialize parameters for relation embedding
    params['W_emb_rel'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(n_rel, emb_dim)).astype('float64'), name='W_emb_rel')

    return params

def char2vec(params,n_char,bias=True):
    '''
    Bi-GRU for encoding input
    '''
    # Variables for positive lhs
    word = T.imatrix() # B x N # input
    mask = T.fmatrix() # B x N # input

    # Variables for negative lhs
    wordn = T.imatrix() # B x N # input
    maskn = T.fmatrix() # B x N # input

    # Input layer over characters
    l_in_source = lasagne.layers.InputLayer(shape=(N_BATCH,None), name='input')

    # Mask layer for variable length sequences
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH,None), name='mask')

    # lookup
    l_clookup_source = lasagne.layers.EmbeddingLayer(l_in_source, input_size=n_char, output_size=CHAR_DIM, W=params['Wc'])

    # f-GRU
    c2w_f_reset = lasagne.layers.Gate(W_in=params['W_c2w_f_r'], W_hid=params['U_c2w_f_r'], W_cell=None, b=params['b_c2w_f_r'], nonlinearity=NL1)
    c2w_f_update = lasagne.layers.Gate(W_in=params['W_c2w_f_z'], W_hid=params['U_c2w_f_z'], W_cell=None, b=params['b_c2w_f_z'], nonlinearity=NL1)
    c2w_f_hidden = lasagne.layers.Gate(W_in=params['W_c2w_f_h'], W_hid=params['U_c2w_f_h'], W_cell=None, b=params['b_c2w_f_h'], nonlinearity=NL2)

    l_fgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_f_reset, updategate=c2w_f_update, hidden_update=c2w_f_hidden, hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=GRAD_CLIP, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # b-GRU
    c2w_b_reset = lasagne.layers.Gate(W_in=params['W_c2w_b_r'], W_hid=params['U_c2w_b_r'], W_cell=None, b=params['b_c2w_b_r'], nonlinearity=NL1)
    c2w_b_update = lasagne.layers.Gate(W_in=params['W_c2w_b_z'], W_hid=params['U_c2w_b_z'], W_cell=None, b=params['b_c2w_b_z'], nonlinearity=NL1)
    c2w_b_hidden = lasagne.layers.Gate(W_in=params['W_c2w_b_h'], W_hid=params['U_c2w_b_h'], W_cell=None, b=params['b_c2w_b_h'], nonlinearity=NL2)

    l_bgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_b_reset, updategate=c2w_b_update, hidden_update=c2w_b_hidden, hid_init=lasagne.init.Constant(0.), backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=GRAD_CLIP, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # Slice final states
    l_f_source = lasagne.layers.SliceLayer(l_fgru_source, -1, 1)
    l_b_source = lasagne.layers.SliceLayer(l_bgru_source, 0, 1)

    # Dense
    l_concat = lasagne.layers.ConcatLayer((l_f_source,l_b_source),axis=1)
    if bias:
        l_c2w_source = lasagne.layers.DenseLayer(l_concat, WDIM, W=params['W_c2w'], b=params['b_c2w'], nonlinearity=NL3)
    else:
        l_c2w_source = lasagne.layers.DenseLayer(l_concat, WDIM, W=params['W_c2w'], b=None, nonlinearity=NL3)

    emb_lhs = lasagne.layers.get_output(l_c2w_source, inputs={l_in_source: word, l_mask: mask})
    emb_lhsn = lasagne.layers.get_output(l_c2w_source, inputs={l_in_source: wordn, l_mask: maskn})
    return word, mask, wordn, maskn, emb_lhs, emb_lhsn, l_c2w_source
    #return word, mask, l_c2w_source # return input variables and output variables

# by Yuxing Zhang
def embedding_rhs(params, n_voc, emb_dim):
    '''
    Embedding part for right hand side entity embedding and right hand side negative entity embedding

    :param params: dict to store parameters
    '''
    # input variables that is right hand side entity
    emb_in_rhs = T.ivector() # B * 1 vector, where each row is a number between 0 and (n_voc - 1) as the index
    emb_in_rhsn = T.ivector() # B * 1 vector, where each row is a number between 0 and (n_voc - 1) as the index

    # Input layer over entity
    l_in_rhs = lasagne.layers.InputLayer(shape=(N_BATCH, ), name = 'rhs_input') # removing input_var to reuse it for negative rhs

    # Embedding layer for rhs entity, and emb_dim should equal # the embedding dimension from RNN model.
    l_emb_rhs = lasagne.layers.EmbeddingLayer(l_in_rhs, input_size=n_voc, output_size=emb_dim, W=params['W_emb_rhs'])

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
