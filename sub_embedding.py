import cPickle as pk
from model import *

if __name__ == "__main__":

    f = open('./data/FB15k_idx2entity.pkl', 'r')
    idx2entity_pre = pk.load(f)
    print len(idx2entity_pre)
    f.close()

    f = open('./data/FB15k_entity2idx.pkl', 'r')
    entity2idx_pre = pk.load(f)
    print len(entity2idx_pre)
    f.close()

    f = open('../js_prescription/data/FB15k_entity2idx.pkl', 'r')
    entity2idx_mer = pk.load(f)
    print len(entity2idx_mer)
    f.close()

    f = open('../js_prescription/data/FB15k_idx2entity.pkl', 'r')
    idx2entity_mer = pk.load(f)
    print len(idx2entity_mer)
    f.close()
    
    idx_list = []
    for i in range(len(idx2entity_pre)):
        entity = idx2entity_pre[i]
        new_idx = entity2idx_mer[entity]
        idx_list += [new_idx]
        #print entity + '\t' + idx2entity_mer[new_idx]
    #print idx_list

    f = open('../js_prescription/FB15k/js_prescription/best_valid_model.pkl', 'r')
    embeddings = pk.load(f)
    leftop = pk.load(f)
    rightop = pk.load(f)
    simfn = pk.load(f)
    print leftop
    print rightop
    print simfn

    #embedding, relationl, relationr = parse_embeddings(embeddings)

    emb = embeddings[0].E.get_value() # numpy matrix
    f.close()
    
    f = open('../prescription/FB15k/prescription/best_valid_model.pkl', 'r')
    embeddings2 = pk.load(f)
    leftop = pk.load(f)
    rightop = pk.load(f)
    simfn = pk.load(f)
    print leftop
    print rightop
    print simfn

    print embeddings[0].E.get_value().shape
    embeddings2[0].E.set_value(emb[:, idx_list])
    print embeddings2[0].E.get_value().shape

    f = open('./FB15k/prescription/best_valid_model_merge.pkl', 'w')
    cPickle.dump(embeddings2, f, -1)
    cPickle.dump(leftop, f, -1)
    cPickle.dump(rightop, f, -1)
    cPickle.dump(simfn, f, -1)
    f.close()


    #print "============ original embedding ============"
    #print emb
    #print "============ original shape ============"
    #print emb.shape


    #new_emb, rl, rr = parse_embeddings(embeddings)
    #print "============ new embedding shape ============"
    #print new_emb.E.get_value().shape
    #print "============ old ============"
    #embedding.E.set_value(emb[:, idx_list])
    #emb2 = embedding.E.get_value()
    #print "============ new embedding ============"
    #print emb2
    #print "============ new shape ============"
    #print emb2.shape

    
