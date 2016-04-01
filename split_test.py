import sys

'''
usage: run this and sort the outputfile, then run this again
'''

def run(filename):
    f = open(filename, 'r')
    dat = f.readlines()
    f.close()
    outputfile = 'data/triples-test-tmp.txt'
    f = open(outputfile, 'w')
    for line in dat:
        tokens = line.strip().split('\t')
        f.write(tokens[1] + '\t' + tokens[0] + '\t' + tokens[2] + '\n')
    f.close()

if __name__ == "__main__":
    run(sys.argv[1])
