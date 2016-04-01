import sys
from random import sample

def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

# split the data into training set, validation set and testing set
def split(path):
    f = open(path + 'triples_freq.txt', 'r')
    dat = f.readlines()
    f.close()
    f_train = open(path + 'prescription-freq-train.txt', 'w')
    f_test = open(path + 'prescription-freq-test.txt', 'w')
    f_valid = open(path + 'prescription-freq-valid.txt', 'w')
    lastdrug = ''
    cur = []
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        if lhs != lastdrug and lastdrug != '':
            valid_idx = []
            test_idx = []
            length = len(cur)
            if length > 3:
                r = sample(range(0, length), 2)
                valid_idx.append(r[0])
                test_idx.append(r[1])
            train_idx = [k for k in range(0, length) if k not in valid_idx and k not in test_idx]
            for j in train_idx:
                f_train.write(cur[j])
            for j in test_idx:
                f_test.write(cur[j])
            for j in valid_idx:
                f_valid.write(cur[j])
            cur = [i]
            lastdrug = lhs
        else:
            cur.append(i)
            lastdrug = lhs
    f_train.close()
    f_test.close()
    f_valid.close()
if __name__ == "__main__":
    split(sys.argv[1])
