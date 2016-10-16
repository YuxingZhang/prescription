import numpy as np
import random
import cPickle as pkl

from collections import OrderedDict

class Batch():
    def __init__(self, lhs, rel, rhs, batch_size=128):
        self.lhs = lhs
        self.rel = rel
        self.rhs = rhs
        self.batch_size = batch_size

        self.prepare()
        self.reset()

    def prepare(self):
        self.indices = np.arange(len(self.lhs))
        self.curr_indices = np.random.permutation(self.indices)

    def reset(self):
        self.curr_indices = np.random.permutation(self.indices)
        self.curr_pos = 0
        self.curr_remaining = len(self.indices)

    def next(self):
        if self.curr_pos >= len(self.indices):
            self.reset()
            raise StopIteration()

        # current batch size
        curr_batch_size = np.minimum(self.batch_size, self.curr_remaining)

        # indices for current batch
        curr_indices = self.curr_indices[self.curr_pos:self.curr_pos+curr_batch_size]
        self.curr_pos += curr_batch_size
        self.curr_remaining -= curr_batch_size

        # data and targets for current batch
        lhs_batch = [self.lhs[ii] for ii in curr_indices]
        rel_batch = [self.rel[ii] for ii in curr_indices]
        rhs_batch = [self.rhs[ii] for ii in curr_indices]

        return lhs_batch, rel_batch, rhs_batch

    def __iter__(self):
        return self

def prepare_data(lhs_b, rel_b, rhs_b, chardict, rel_dict, rhs_dict, n_chars, use_beos=False):
    """
    Prepare the data for training - add masks and remove infrequent characters
    """
    lhs_idx = []
    for cc in lhs_b:
        current = list(cc)
        lhs_idx.append([chardict[c] if c in chardict and chardict[c] <= n_chars else 0 for c in current])

    rel_idx = [rel_dict[yy] for yy in rel_b] # convert each relation to its index
    rhs_idx = [rhs_dict[yy] for yy in rhs_b] # convert each right hand side to its index

    len_lhs = [len(s) for s in lhs_idx]
    max_length = max(len_lhs)
    n_samples = len(lhs_idx)

    lhs_in = np.zeros((n_samples,max_length)).astype('int32')
    lhs_mask = np.zeros((n_samples,max_length)).astype('float32')
    rel_in = np.zeros((n_samples)).astype('int32')
    rhs_in = np.zeros((n_samples)).astype('int32')

    # random index as the negative triples
    rhsn_in = np.random.randint(len(rhs_dict), size=n_samples).astype('int32')
    
    for idx, lhs_idx_i in enumerate(lhs_idx):
        lhs_in[idx,:len_lhs[idx]] = lhs_idx_i
        lhs_mask[idx,:len_lhs[idx]] = 1.
        rel_in[idx] = rel_idx[idx]
        rhs_in[idx] = rhs_idx[idx]

    return lhs_in, lhs_mask, rel_in, rhs_in, rhsn_in

def build_char_dictionary(text):
    """
    Build a character dictionary
    """
    charcount = OrderedDict()
    for cc in text:
        chars = list(cc)
        for c in chars:
            if c not in charcount:
                charcount[c] = 0
            charcount[c] += 1
    chars = charcount.keys()
    freqs = charcount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    chardict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        chardict[chars[sidx]] = idx + 1

    #print charcount # TODO why sort it??
    #print chardict

    return chardict, charcount

def build_entity_dictionary(targets):
    """
    Build a label dictionary
    """
    labelcount = OrderedDict()
    for l in targets:
        if l not in labelcount:
            labelcount[l] = 0
        labelcount[l] += 1
    labels = labelcount.keys()
    freqs = labelcount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    labeldict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        labeldict[labels[sidx]] = idx

    return labeldict, labelcount

def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'w') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)

def load_labeled_entities(f): # split each line into lhs, rel and rhs
    lhs = []
    rel = []
    rhs = []
    for line in f:
        entities = line.rstrip().split('\t')
        lhs.append(entities[0])
        rel.append(entities[1])
        rhs.append(entities[2])
    return lhs, rel, rhs
