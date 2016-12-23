# split the data into training set, validation set and testing set randomly
import sys
from random import sample
import random

def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

def split(path):
    f = open(path + 'triples.txt', 'r')
    dat = f.readlines()
    f.close()
    f_train = open(path + 'freebase_mtr100_mte100-train.txt', 'w')
    f_test = open(path + 'freebase_mtr100_mte100-test.txt', 'w')
    f_valid = open(path + 'freebase_mtr100_mte100-valid.txt', 'w')
    test_size = 0.1
    valid_size = 0.1
    idx = sample(range(0, len(dat)), int((test_size + valid_size) * len(dat)))
    print len(dat)
    print len(idx)
    test_idx = idx[: int(len(idx) / 2)]
    valid_idx = idx[(len(idx) / 2):]
    #train_idx = [k for k in range(0, len(dat)) if k not in idx]
    for j in range(0, len(dat)):
        if j not in idx:
            f_train.write(dat[j])
    for j in test_idx:
        f_test.write(dat[j])
    for j in valid_idx:
        f_valid.write(dat[j])

    f_train.close()
    f_test.close()
    f_valid.close()
if __name__ == "__main__":
    #split(sys.argv[1])
    split('data/')
