import cPickle as pk
from model import *

if __name__ == "__main__":

    f = open('./FB15k/prescription/best_valid_model.pkl')
    embeddings = pk.load(f)
    embedding, relationl, relationr = parse_embeddings(embeddings)
    emb = embedding.E.get_value() # numpy matrix
    f.close()

    print "============ original embedding ============"
    print emb
    print "============ original shape ============"
    print emb.shape

    embedding.E.set_value(emb[1:3, :])
    emb2 = embedding.E.get_value()
    print "============ new embedding ============"
    print emb2
    print "============ new shape ============"
    print emb2.shape
