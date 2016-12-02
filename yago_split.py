# split the YAGOFact data set into training set, validation set and test set
# sparse: train:valid:test = 80%:10%:10%
import sys
import numpy as np

# split the YAGOFact data into training set, validation set and testing set
def split_fn():
    left_dict = {}
    rel_dict = {}
    right_dict = {}
    f = open('/remote/curtis/wcohen/data/ontological-pathfinding/YAGOFacts.csv', 'r')
    dat = f.readlines()
    subsample_fraction = 0.01
    train_pr = 0.8
    valid_pr = 0.1
    f.close()
    f_train = open('data/yago-sparse-train.txt', 'w')
    f_test = open('data/yago-sparse-test.txt', 'w')
    f_valid = open('data/yago-sparse-valid.txt', 'w')
    train_set = []
    valid_set = []
    test_set = []
    for i in dat:
        words = i.upper().strip().split(' ')
        if len(words) != 3:
            continue
        left = words[1]
        rel = words[0]
        right = words[2]
        left_dict[left] = 1
        rel_dict[rel] = 1
        right_dict[right] = 1
        words = left + '\t' + rel + '\t' + right + '\n'
        r = np.random.rand()

        # subsample the data
        if r > subsample_fraction:
            continue

        r = np.random.rand()
        if r < train_pr:
            train_set += [words]
        elif r < (train_pr + valid_pr):
            valid_set += [words]
        else:
            test_set += [words]
    f_train.writelines(train_set)
    f_valid.writelines(valid_set)
    f_test.writelines(test_set)
    f_train.close()
    f_test.close()
    f_valid.close()
    print 'Left entities: {}, relations: {}, right entities: {}'.format(len(left_dict), len(rel_dict), len(right_dict))
if __name__ == "__main__":
    split_fn()
