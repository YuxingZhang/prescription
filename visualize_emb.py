import cPickle as pk
from model import *

if __name__ == "__main__":
    f = open('./data/FB15k_entity2idx.pkl')
    ent2idx = pk.load(f) # a dic from entity to index
    f.close()
    f = open('./data/FB15k_idx2entity.pkl')
    idx2ent = pk.load(f) # a dic from entity to index
    f.close()

    f = open('./FB15k/prescription/best_valid_model.pkl')
    embeddings = pk.load(f)
    embedding, relationl, relationr = parse_embeddings(embeddings)
    embedding = embedding.E.get_value()
    f.close()
    
    shape = embedding.shape
    f = open('embedding.txt', 'w')
    for i in range(shape[1]):
        for j in range(shape[0]):
            f.write(str(embedding[j][i]) + ' ')
        f.write('\n')
    f.close()
