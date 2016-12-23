# input: training set, validation set, test set
# output: part of test set such that each left hand side entity appear no more
# than FREQ times in the training set.

import sys
import io
import numpy as np
import batch

# split the data into training set, validation set and testing set
if __name__ == "__main__":
    max_freq = 2 # selecting lhs that appear <= max_freq times in the training set
    lhs, rel, rhs = batch.load_labeled_entities(io.open("../data/prescription-sparse2-train.txt"))
    lhs_dict, lhs_count = batch.build_entity_dictionary(lhs)

    lhs, rel, rhs = batch.load_labeled_entities(io.open("../data/prescription-sparse2-test.txt"))
    buf = []
    for i in range(len(lhs)):
        if lhs[i] not in lhs_count or lhs_count[lhs[i]] <= max_freq:
            buf.append("{}\t{}\t{}\n".format(lhs[i], rel[i], rhs[i]))

    lhs, rel, rhs = batch.load_labeled_entities(io.open("../data/prescription-sparse2-valid.txt"))
    for i in range(len(lhs)):
        if lhs[i] not in lhs_count or lhs_count[lhs[i]] <= max_freq:
            buf.append("{}\t{}\t{}\n".format(lhs[i], rel[i], rhs[i]))

    with open("../data/prescription-sparse2-rare-{}-test.txt".format(max_freq), "w") as f_out:
        f_out.writelines(buf)
