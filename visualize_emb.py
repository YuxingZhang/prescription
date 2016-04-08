import cPickle as pk

if __name__ == "__main__":
    f = open('./data/FB15k_entity2idx.pkl')
    ent2idx = pk.load(f) # a dic from entity to index
    f.close()
    f = open('./data/FB15k_idx2entity.pkl')
    idx2ent = pk.load(f) # a dic from entity to index
    f.close()

    f = open('./FB15k/prescription/best_valid_model.pkl')
    embedding = pk.load(f)

