import sys

def run():
    filename = '../data/prescription-freq-test.txt'
    f = open(filename, 'r')
    dat = f.readlines()
    f.close()
    f = open(filename, 'w')
    for line in dat:
        tokens = line.strip().split('\t')
        f.write(tokens[1] + '\t' + tokens[0] + '\t' + tokens[2] + '\n')
    f.close()

if __name__ == "__main__":
    run()
