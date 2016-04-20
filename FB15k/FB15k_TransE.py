#! /usr/bin/python
from FB15k_exp import *

#totepochs = 500

#launch(op='TransE', simfn='L2', ndim=50, nhid=50, marge=0.5, lremb=0.01, lrparam=0.01, nbatches=100, totepochs=500, test_all=10, neval=1000, savepath='FB15k_TransE', datapath='../data/prescription/')

launch(datapath='../data/', Nent=3747, rhoE=1, rhoL=5,
        Nsyn=3743, Nrel=4, loadmodel=False, loademb=False, op='TransE',
        simfn='L2', ndim=50, nhid=50, marge=0.5, lremb=0.01, lrparam=0.01,
        nbatches=100, totepochs=300, test_all=10, neval=400, seed=123,
        savepath='prescription')
