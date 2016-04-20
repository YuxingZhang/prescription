import cPickle as pk
from model import *

if __name__ == "__main__":

    f = open('./FB15k/prescription/best_valid_model.pkl')
    embeddings = pk.load(f)
    embedding, relationl, relationr = parse_embeddings(embeddings)
    embedding = embedding.E.get_value() # numpy matrix
    f.close()

    print embedding
