''' extract the embeddings from the best_valid_model and visualize a test file by computing
    rhs - lhs and visualize it using tsne
    usage: python visualize_emb.py <input_test_file_name> <output_file_name>
'''
import cPickle as pk
from model import *

def load_emb(filename):
    '''load embedding from files, including embedding, relationl, relationr'''
    f = open(filename)
    embeddings = pk.load(f)
    f.close()
    embedding, relationl, relationr = parse_embeddings(embeddings)
    embedding = np.transpose(embedding.E.get_value())
    relationl = np.transpose(relationl.E.get_value())
    relationr = np.transpose(relationr.E.get_value())
    return embedding, relationl, relationr

def load_map(filename):
    '''load idx2entity or entity2idx from files'''
    f = open(filename)
    mapping = pk.load(f) # a dic from entity to index
    f.close()
    return mapping

def print_emb(embedding):
    shape = embedding.shape
    f = open('embedding.txt', 'w')
    for i in range(shape[1]):
        for j in range(shape[0]):
            f.write(str(embedding[j][i]) + ' ')
        f.write('\n')
    f.close()

def parseline(line):
    lhs, rel, rhs = line.strip().split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

if __name__ == "__main__":
    ent2idx = load_map('./data/FB15k_entity2idx.pkl') # a dic from entity to index
    idx2ent = load_map('./data/FB15k_idx2entity.pkl') # a dic from idx to entity
    embedding, relationl, relationr = load_emb('./FB15k/prescription/best_valid_model.pkl')

    test_file = sys.argv[1] # test file name
    output_file = sys.argv[2] # test file name
    f = open(test_file, 'r')
    data = f.readlines()
    f.close()
    f = open(output_file, 'w')
    for line in data:
        lhs, rel, rhs = parseline(line)
        lidx = ent2idx[lhs[0]]
        ridx = ent2idx[rhs[0]]
        lemb = embedding[lidx, :]
        remb = embedding[ridx, :]
        dif = remb - lemb
        for i in range(len(dif)):
            f.write(' ' + str(dif[i]))
        f.write('\n')
