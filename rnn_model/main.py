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
import evaluate
from settings import N_BATCH, N_EPOCH, DISPF, SAVEF, VALF
from model import charLM

T1 = 0.01
T2 = 0.001

if __name__=='__main__':
    save_path = sys.argv[-1]
    shutil.copyfile('settings.py','%s/settings.txt'%save_path)

    print("Preparing data...")
    # load
    lhs, rel, rhs = batch.load_labeled_entities(io.open(sys.argv[1],'r'))
    lhs_v, rel_v, rhs_v = batch.load_labeled_entities(io.open(sys.argv[2],'r'))

    # dictionaries
    chardict, charcount = batch.build_char_dictionary(lhs)
    n_char = len(chardict.keys()) + 1
    batch.save_dictionary(chardict,charcount,'%s/dict.pkl' % save_path)

    #labeldict, labelcount = batch.build_entity_dictionary(y)
    # build dictionary for relations
    rel_dict, rel_count = batch.build_entity_dictionary(rel)
    batch.save_dictionary(rel_dict, rel_count, '%s/rel_dict.pkl' % save_path)
    n_rel = len(rel_dict.keys())
    #print relcount  # this tells number of triples in different relations

    # build dictionary for right hand side entities
    rhs_dict, rhs_count = batch.build_entity_dictionary(rhs)
    batch.save_dictionary(rhs_dict, rhs_count, '%s/rhs_dict.pkl' % save_path)
    n_voc = len(rhs_dict.keys())

    # batches
    train_iter = batch.Batch(lhs, rel, rhs, batch_size=N_BATCH)
    valid_iter = batch.Batch(lhs_v, rel_v, rhs_v, batch_size=N_BATCH)

    print("Building model...")
    m = charLM(n_char, n_voc, n_rel) # emb_dim = WDIM by default

    # Training
    print("Training...")
    uidx = 0
    start = time.time()
    valcosts = []
    max_acc = 0.
    try:
	for epoch in range(N_EPOCH):
	    n_samples = 0
            train_cost = 0.
	    print("Epoch {}".format(epoch))

            # updating learning rate or stop iteration
            # learning schedule
            if len(valcosts) > 1:
                change = (valcosts[-1]-valcosts[-2])/abs(valcosts[-2])
                if change < T1:
                    print("Updating Schedule...")
                    m.update_learningrate()
                    T1 = T1/2
            # stopping criterion
            if len(valcosts) > 4:
                deltas = []
                for i in range(3):
                    deltas.append((valcosts[-i-1]-valcosts[-i-2])/abs(valcosts[-i-2]))
                if all([d < T2 for d in deltas]):
                    break

            # train
            ud_start = time.time()
	    for lhs_b, rel_b, rhs_b in train_iter: # one batch
		n_samples += len(lhs_b)
		uidx += 1
		lhs_in, lhs_mask, rel_in, rhs_in, rhsn_in = batch.prepare_data(lhs_b, rel_b, rhs_b, chardict, rel_dict, rhs_dict, n_chars=n_char)
                #print lhs_in.shape print lhs_mask.shape print rel_in.shape print rhs_in.shape print rhsn_in.shape

		curr_cost = m.train(lhs_in, lhs_mask, rel_in, rhs_in, rhsn_in)
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
                # validation
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
                        lhs_v, lhs_vmask, rel_v, rhs_v, rhsn_v = batch.prepare_data(lhs_vb, rel_vb, rhs_vb, chardict, rel_dict, rhs_dict, n_chars=n_char)
                        valid_mean_rank += m.rank_right(lhs_v, lhs_vmask, rel_v, rhs_v)

                        #vc = m.validate(x,x_m,y)
                        #validation_cost += vc*x.shape[0]
                        #n_val_samples += x.shape[0]
                    #R = evaluate.Result.fromlists(np.asarray(val_pred), val_targets, n_classes)
                    #if R.acc > max_acc:
                    #    max_acc = R.acc
                    #    m.save_model('%s/best_model.npz' % save_path)
                    print("Epoch {} Update {} Training Cost {} Validation mean rank {}".format(epoch, 
                        uidx, train_cost/n_samples, np.mean(np.array(valid_mean_rank))))
                    #valcosts.append(R.acc)

            print("Saving...")
            m.save_model('%s/model_%d.npz' % (save_path,epoch))

    except KeyboardInterrupt:
	pass
    print("Total training time = {}".format(time.time()-start))
