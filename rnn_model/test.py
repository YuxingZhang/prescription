import numpy as np
import theano
import theano.tensor as T
import sys
import lasagne
import time
import cPickle as pkl
import io

import batch
from model import charLM
import evaluate

def main(args):
    data_path = args[0]
    model_path = args[1]
    save_path = args[2]
    if len(args)>3:
        m_num = int(args[3])

    print("Preparing Data...")
    # Test data
    Xt,yt = batch.load_labeled_entities(io.open(data_path, 'r'))

    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = len(labeldict.keys())

    # iterators
    test_iter = batch.Batch(Xt, yt, batch_size=256)

    # Model
    print("Loading model params...")
    if len(args)>3:
        m = charLM(n_char, n_classes, '%s/model_%d.npz'%(model_path,m_num))
    else:
        m = charLM(n_char, n_classes, '%s/model.npz'%model_path)

    # Test
    print("Testing...")
    out_data = []
    out_pred = []
    out_emb = []
    out_target = []
    for xr,yr in test_iter:
        x, x_m, y = batch.prepare_data(xr, yr, chardict, labeldict, n_chars=n_char)
        p = m.predict(x,x_m)
        e = m.encode(x,x_m)
        ranks = np.argmax(p, axis=1)

        for idx, item in enumerate(xr):
            out_data.append(item)
            out_pred.append(ranks[idx])
            out_emb.append(e[idx,:])
            out_target.append(y[idx])

    # Save
    print("Saving...")
    with open('%s/data.pkl'%save_path,'w') as f:
        pkl.dump(out_data,f)
    with open('%s/predictions.npy'%save_path,'w') as f:
        np.save(f,np.asarray(out_pred))
    with open('%s/embeddings.npy'%save_path,'w') as f:
        np.save(f,np.asarray(out_emb))
    with open('%s/targets.pkl'%save_path,'w') as f:
        pkl.dump(out_target,f)


if __name__ == '__main__':
    main(sys.argv[1:])
    evaluate.main(sys.argv[3],sys.argv[2])
