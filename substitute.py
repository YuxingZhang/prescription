# read triples, and substitute the relations with a unique name of that 
# relation to make it different from other entities
import sys

if __name__ == "__main__":
    in_file = sys.argv[1]
    #out_file = sys.argv[2]
    out_file = in_file
    f = open(in_file, 'r')
    lines = f.readlines()
    f.close()
    f = open(out_file, 'w')
    for i in lines:
        tokens = i.strip().split('\t')
        if len(tokens) < 3
            continue
        f.write(tokens[0])
        f.write('\t')
        f.write(tokens[1] + "_REL")
        f.write('\t')
        f.write(tokens[2])
        f.write('\n')
    f.close()
