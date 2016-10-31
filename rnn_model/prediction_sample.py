# load the trained model and compute prediction examples given a test data set

import numpy as np
import shutil
import theano
import theano.tensor as T
import sys
import lasagne
import time
import cPickle as pkl
import io

from collections import OrderedDict

import batch
from settings import N_BATCH, N_EPOCH, DISPF, SAVEF, VALF, WDIM
from model_transe import charLM                   # TODO change model
from model_transe import load_params_shared       # TODO change model

model = "_tr" # or "" or "_nn" where nn is hybrid   # TODO change model

if __name__ == "__main__":
    lhs, rel, rhs = batch.load_labeled_entities(io.open("../data/prescription-sparse2-train.txt")) # sparse 2 is by different train,valid,test ratio
    chardict, charcount = batch.build_char_dictionary(lhs)
    n_char = len(chardict.keys()) + 1
    lhs_dict, lhs_count = batch.build_entity_dictionary(lhs)
    n_lhs = len(lhs_dict.keys())
    rel_dict, rel_count = batch.build_entity_dictionary(rel)
    print rel_dict
    n_rel = len(rel_dict.keys())
    rhs_dict, rhs_count = batch.build_entity_dictionary(rhs)
    n_rhs = len(rhs_dict.keys())

    lhs_s, rel_s, rhs_s = batch.load_labeled_entities(io.open("../data/prescription-sparse2-rare-2-test.txt"))
    test_iter = batch.Batch(lhs_s, rel_s, rhs_s, batch_size=N_BATCH)

    m = charLM(n_char, n_lhs + 1, n_rel, n_rhs) # emb_dim = WDIM by default
    m.param = load_params_shared("temp{}/best_model.npz".format(model))

    # compute example predictions 
    m.compute_emb_right_all()
    mean_rank = []
    X_vis = np.zeros((1, WDIM))
    Y_vis = []
    for lhs_sb, rel_sb, rhs_sb in test_iter: # one batch
        lhs_s, rel_s, rhs_s = \
                batch.prepare_vs_tr(lhs_sb, rel_sb, rhs_sb, chardict, lhs_dict, rel_dict, rhs_dict, n_chars=n_char) # TODO change model
        test_mean_rank = m.rank_right(lhs_s, rel_s, rhs_s)
        mean_rank += test_mean_rank

        print "======"
        print (m.pred_rel_fn(lhs_s, rhs_s)).shape
        print X_vis.shape
        X_vis = np.concatenate(X_vis, m.pred_rel_fn(lhs_s, rhs_s))
        Y_vis += [(str(rel_dict[ii]) + '\n') for ii in rel_s]
        
        '''
        for i in range(len(test_mean_rank)):
            if test_mean_rank[i] < 10.0:
                top_scored_rhs = m.top_scored_rhs([lhs_s[i]], [lhs_smask[i]], [lhs_emb_s[i]], [rel_s[i]], 12) # the indices of top scored rhs
                tops = []
                for j in top_scored_rhs:
                    tops += [rhs_dict.keys()[j]]
                print "Good predict: lhs={}, rel={}, rhs={}, rank={}, top={}".format(lhs_sb[i], rel_sb[i], rhs_sb[i], test_mean_rank[i], tops)
        '''

    # save visualization X and Y
    X_vis = X_vis[1:, :]
    np.savetxt("temp{}/X_vis.txt", X_vis, delimiter=' ')
    with open("temp{}/Y_vis.txt") as y_out:
        y_out.writelines(Y_vis)

    print "Mean rank: {}, rank: {}".format(sum(mean_rank) / float(len(mean_rank)), mean_rank)
