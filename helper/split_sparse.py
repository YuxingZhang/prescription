# split the unique prescription data set into training set, validation set and test set
# sparse: train:valid:test = 90%:5%:5%
# sparse2: train:valid:test = 80%:10%:10%
import sys
import numpy as np

# split the data into training set, validation set and testing set
def split(path):
    f = open(path + 'full-triples-unique.txt', 'r')
    dat = f.readlines()
    train_pr = 0.8
    valid_pr = 0.1
    f.close()
    f_train = open(path + 'prescription-sparse2-train.txt', 'w')
    f_test = open(path + 'prescription-sparse2-test.txt', 'w')
    f_valid = open(path + 'prescription-sparse2-valid.txt', 'w')
    train_set = []
    valid_set = []
    test_set = []
    for i in dat:
        words = i.strip().split('\t')
        if len(words) != 3:
            continue
        r = np.random.rand()
        if r < train_pr:
            train_set += [i]
        elif r < (train_pr + valid_pr):
            valid_set += [i]
        else:
            test_set += [i]
    f_train.writelines(train_set)
    f_valid.writelines(valid_set)
    f_test.writelines(test_set)
    f_train.close()
    f_test.close()
    f_valid.close()
if __name__ == "__main__":
    split(sys.argv[1])
