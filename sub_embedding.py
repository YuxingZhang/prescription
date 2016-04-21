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
    print '===='
    for i in range(len(idx2entity_mer)):
        if idx2entity_mer[i] not in entity2idx_mer:
            print "not contained", idx2entity_mer[i]
    for i in range(len(idx2entity_mer) - 10, len(idx2entity_mer)):
        print idx2entity_mer[i]
    print '===='
    rels = ['CONSUMED_IN', 'CONTAINS_INGREDIENT', 'DOSAGE_FORM', 'PACKAGE_FORM']
    for i in rels:
        print entity2idx_mer[i]
    print '===='
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
    embeddings1 = pk.load(f)
    leftop1 = pk.load(f)
    rightop1 = pk.load(f)
    simfn1 = pk.load(f)
    print embeddings1[0].E.get_value()[1:5, -4:]
    print embeddings1[1].E.get_value()[1:5, -4:]
    print embeddings1[2].E.get_value()[1:5, -4:]

    #embedding, relationl, relationr = parse_embeddings(embeddings)

    emb = embeddings1[0].E.get_value() # numpy matrix
    print "rel size"
    print embeddings1[1].E.get_value().shape
    print embeddings1[2].E.get_value().shape
    f.close()
    
    f = open('../prescription/FB15k/prescription/best_valid_model.pkl', 'r')
    embeddings2 = pk.load(f)
    print "==2=="
    print embeddings2[0].E.get_value()[1:5, -4:]
    print embeddings2[1].E.get_value()[1:5, -4:]
    print embeddings2[2].E.get_value()[1:5, -4:]
    leftop2 = pk.load(f)
    rightop2 = pk.load(f)
    simfn2 = pk.load(f)
    print embeddings2[1].E.get_value().shape
    print embeddings2[2].E.get_value().shape

    new_embedding_entity = emb[:, idx_list]
    print "rel shape", embeddings1[1].E.get_value().shape
    new_embedding_rel = embeddings1[1].E.get_value()[:, -10: -6]

    #print embeddings1[0].E.get_value().shape
    embeddings1[0].E.set_value(new_embedding_entity)
    #print embeddings1[0].E.get_value().shape
    embeddings2[0].E.set_value(new_embedding_entity)
    embeddings2[1].E.set_value(new_embedding_rel)
    embeddings2[2].E.set_value(new_embedding_rel)

    open('./FB15k/prescription/best_valid_model_merge.pkl', 'w').close()
    f = open('./FB15k/prescription/best_valid_model_merge.pkl', 'w')
    cPickle.dump(embeddings2, f, -1)
    cPickle.dump(leftop2, f, -1)
    cPickle.dump(rightop2, f, -1)
    cPickle.dump(simfn2, f, -1)
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

    
