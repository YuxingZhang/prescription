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
from settings import N_BATCH, N_EPOCH, DISPF, SAVEF, VALF
from model_nn import charLM

T1 = 0.01
T2 = 0.001

if __name__=='__main__':
    save_path = sys.argv[-1]
    shutil.copyfile('settings.py','%s/settings.txt'%save_path)

    print("Preparing data...")
    # load
    #lhs, rel, rhs = batch.load_labeled_entities(io.open(sys.argv[1],'r'))
    #lhs_v, rel_v, rhs_v = batch.load_labeled_entities(io.open(sys.argv[2],'r'))
    lhs, rel, rhs = batch.load_labeled_entities(io.open("../data/prescription-freq-train.txt"))
    lhs_v, rel_v, rhs_v = batch.load_labeled_entities(io.open("../data/prescription-freq-valid.txt"))
    lhs_s, rel_s, rhs_s = batch.load_labeled_entities(io.open("../data/prescription-freq-test.txt"))

    # left hand side dictionaries, both character and entity
    chardict, charcount = batch.build_char_dictionary(lhs)
    n_char = len(chardict.keys()) + 1
    batch.save_dictionary(chardict,charcount,'%s/dict.pkl' % save_path)

    lhs_dict, lhs_count = batch.build_entity_dictionary(lhs)
    n_lhs = len(lhs_dict.keys())
    batch.save_dictionary(lhs_dict,lhs_count,'%s/lhs_dict.pkl' % save_path)

    # build dictionary for relations
    rel_dict, rel_count = batch.build_entity_dictionary(rel)
    batch.save_dictionary(rel_dict, rel_count, '%s/rel_dict.pkl' % save_path)
    n_rel = len(rel_dict.keys())
    # this tells number of triples in different relations

    # build dictionary for right hand side entities
    rhs_dict, rhs_count = batch.build_entity_dictionary(rhs)
    batch.save_dictionary(rhs_dict, rhs_count, '%s/rhs_dict.pkl' % save_path)
    n_rhs = len(rhs_dict.keys())

    # batches
    train_iter = batch.Batch(lhs, rel, rhs, batch_size=N_BATCH)
    valid_iter = batch.Batch(lhs_v, rel_v, rhs_v, batch_size=N_BATCH)
    test_iter = batch.Batch(lhs_s, rel_s, rhs_s, batch_size=N_BATCH)

    print("Building model...")
    m = charLM(n_char, n_lhs + 1, n_rel, n_rhs) # emb_dim = WDIM by default

    # Training
    print("Training...")
    uidx = 0
    start = time.time()
    valcosts = []
    min_mean_rank = float("inf")
    try:
	for epoch in range(N_EPOCH):
	    n_samples = 0
            train_cost = 0.
	    print("Epoch {}".format(epoch))

            # updating learning rate or stop iteration
            # learning schedule
            if len(valcosts) > 1:
                change = (valcosts[-2]-valcosts[-1])/abs(valcosts[-2])
                print "change = {}".format(change)
                if change < T1:
                    print("Updating Schedule...")
                    m.update_learningrate()
                    T1 = T1/2
            # stopping criterion
            #if len(valcosts) > 4:
            #    deltas = []
            #    for i in range(3):
            #        deltas.append((valcosts[-i-1]-valcosts[-i-2])/abs(valcosts[-i-2]))
            #    if all([d < T2 for d in deltas]):
            #        break

            # train
            ud_start = time.time()
	    for lhs_b, rel_b, rhs_b in train_iter: # one batch
		n_samples += len(lhs_b)
		uidx += 1
		lhs_in, lhs_mask, lhsn_in, lhsn_mask, in_emb_lhs, in_emb_lhsn, rel_in, rhs_in, rhsn_in = \
                        batch.prepare_data_nn(lhs_b, rel_b, rhs_b, chardict, lhs_dict, rel_dict, rhs_dict, n_chars=n_char)
                ''' Check shapes
                print lhs_in.shape 
                print lhs_mask.shape 
                print lhsn_in.shape 
                print lhsn_mask.shape 
                print in_emb_lhs
                print in_emb_lhsn
                print rel_in.shape 
                print rhs_in.shape 
                print rhsn_in.shape
                '''
		curr_cost = m.train(lhs_in, lhs_mask, lhsn_in, lhsn_mask, in_emb_lhs, in_emb_lhsn, rel_in, rhs_in, rhsn_in)
                train_cost += curr_cost * len(lhs_b) # why times length, because the training function returns the mean
		ud = time.time() - ud_start
                
                # detect nan cost
		if np.isnan(curr_cost) or np.isinf(curr_cost):
		    print("Nan detected.")
                    sys.exit()
                # display information
		if np.mod(uidx, DISPF) == 0:
		    print("Epoch {} Update {} Cost {} Time {} Samples {}".format(epoch,uidx,curr_cost,ud,len(lhs_b)))
                # save model
		if np.mod(uidx,SAVEF) == 0:
		    print("Saving...")
                    m.save_model('%s/model.npz' % save_path)
                # validation and testing
                if np.mod(uidx,VALF) == 0:
                    print("Testing on Validation set...")
                    val_pred = []
                    val_targets = []
                    validation_cost = 0.
                    n_val_samples = 0
                    #for xr,yr in valid_iter:
                    valid_mean_rank = []

                    # compute right hand side embeddings for all entities using the new parameters
                    m.compute_emb_right_all()
                    for lhs_vb, rel_vb, rhs_vb in valid_iter: # one batch
                        lhs_v, lhs_vmask, lhs_emb_v, rel_v, rhs_v = \
                                batch.prepare_vs_nn(lhs_vb, rel_vb, rhs_vb, chardict, lhs_dict, rel_dict, rhs_dict, n_chars=n_char)
                        valid_mean_rank += m.rank_right(lhs_v, lhs_vmask, lhs_emb_v, rel_v, rhs_v)
                    valid_mean_rank = np.array(valid_mean_rank)

                    cur_mean_rank = np.mean(valid_mean_rank)
                    if cur_mean_rank < min_mean_rank:
                        min_mean_rank = cur_mean_rank
                        m.save_model('%s/best_model.npz' % save_path)
                        print "####### NEW BEST VALID #######"

                    test_mean_rank = []
                    for lhs_sb, rel_sb, rhs_sb in test_iter: # one batch
                        lhs_s, lhs_smask, lhs_emb_s, rel_s, rhs_s = \
                                batch.prepare_vs_nn(lhs_sb, rel_sb, rhs_sb, chardict, lhs_dict, rel_dict, rhs_dict, n_chars=n_char)
                        test_mean_rank += m.rank_right(lhs_s, lhs_smask, lhs_emb_s, rel_s, rhs_s)
                    test_mean_rank = np.array(test_mean_rank)

                    print("Epoch {} Update {} Training Cost {} Validation mean rank {} Validation hit@10 {}% Validation hit@1 {}%".format(epoch, 
                        uidx, train_cost/n_samples, cur_mean_rank, float(100 * sum(valid_mean_rank < 10)) / float(valid_mean_rank.shape[0]),
                        float(100 * sum(valid_mean_rank < 1)) / float(valid_mean_rank.shape[0])))
                    print("                                               Testing mean rank {} Testing hit@10 {}% Testing hit@1 {}%".format(
                        np.mean(test_mean_rank), float(100 * sum(test_mean_rank < 10)) / float(test_mean_rank.shape[0]),
                        float(100 * sum(test_mean_rank < 1)) / float(test_mean_rank.shape[0])))
                    valcosts.append(cur_mean_rank) # TODO add stop criteria

            print("Saving...")
            m.save_model('%s/model_%d.npz' % (save_path,epoch))

    except KeyboardInterrupt:
	pass
    print("Total training time = {}".format(time.time()-start))
    m.compute_emb_right_all()
    for lhs_sb, rel_sb, rhs_sb in test_iter: # one batch
        lhs_s, lhs_smask, lhs_emb_s, rel_s, rhs_s = \
                batch.prepare_vs_nn(lhs_sb, rel_sb, rhs_sb, chardict, lhs_dict, rel_dict, rhs_dict, n_chars=n_char)
        test_mean_rank = m.rank_right(lhs_s, lhs_smask, lhs_emb_s, rel_s, rhs_s)
        for i in range(len(test_mean_rank)):
            if test_mean_rank[i] < 3.0:
                top_scored_rhs = m.top_scored_rhs([lhs_s[i]], [lhs_smask[i]], [lhs_emb_s[i]], [rel_s[i]], 5) # the indices of top scored rhs
                print "Good predict: lhs={}, rel={}, rhs={}, rank={}, top={}".format(lhs_sb[i], rel_sb[i], rhs_sb[i], test_mean_rank[i], top_scored_rhs)
