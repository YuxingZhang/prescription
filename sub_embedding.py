import cPickle as pk
from model import *

if __name__ == "__main__":

    f = open('./data/FB15k_idx2entity.pkl')
    idx2entity_pre = pk.load(f)
    f.close()

    f = open('../js_prescription/data/FB15k_entity2idx.pkl')
    entity2idx_mer = pk.load(f)
    f.close()
    
    idx_list = []
    for i in range(len(idx2entity_pre)):
        entity = idx2entity_pre[i]
        idx_list += [entity2idx_mer[entity]]

    f = open('../js_prescription/FB15k/js_prescription/best_valid_model.pkl')
    embeddings = pk.load(f)
    embedding, relationl, relationr = parse_embeddings(embeddings)
    emb = embedding.E.get_value() # numpy matrix
    f.close()

    print "============ original embedding ============"
    print emb
    print "============ original shape ============"
    print emb.shape

    embedding.E.set_value(emb[:, idx_list])
    emb2 = embedding.E.get_value()
    print "============ new embedding ============"
    print emb2
    print "============ new shape ============"
    print emb2.shape
