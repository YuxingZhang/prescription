import sys
import numpy as np
import cPickle as pkl
import io

class Result():
    def __init__(self, predictions, targets, nclasses):
        self.preds = predictions
        self.targets = targets
        self.nclasses = nclasses
        self.CM = self.conf_matrix()
        self.precision, self.recall, self.fscore = self.scores()
        self.macroF = np.mean(self.fscore)
        self.acc = self.accuracy()

    @classmethod
    def fromfile(cls, result_path, dict_path):
        with open('%s/predictions.npy'%result_path,'r') as f:
            preds = np.load(f)
        with open('%s/targets.pkl'%result_path,'r') as f:
            targets = pkl.load(f)
        with open('%s/data.pkl'%result_path,'r') as f:
            cls.data = pkl.load(f)
        with open('%s/embeddings.npy'%result_path,'r') as f:
            cls.emb = np.load(f)
        with open('%s/label_dict.pkl'%dict_path,'r') as f:
            cls.labeldict = pkl.load(f)

        cls.inverse_labeldict = cls.invert(cls.labeldict)

        nclasses = len(cls.labeldict.keys())
        return cls(preds, targets, nclasses)

    @classmethod
    def fromlists(cls, preds, targets, nclasses):
        return cls(preds, targets, nclasses)

    @staticmethod
    def invert(labeldict):
        inverse = {}
        for kk,vv in labeldict.iteritems():
            inverse[vv] = kk
        return inverse

    def conf_matrix(self):
        C = np.zeros((self.nclasses,self.nclasses))
        for idx, p in enumerate(self.preds):
            t = self.targets[idx]
            C[t,p] = C[t,p] + 1
        return C

    def accuracy(self):
        return float(np.trace(self.CM))/float(np.sum(self.CM))

    def cohens_kappa(self):
        p_obs = float(np.trace(self.CM))/float(np.sum(self.CM))
        row_sums = np.sum(self.CM, axis=1)
        col_sums = np.sum(self.CM, axis=0)
        p_exp = float(np.sum(row_sums*col_sums))/float(np.sum(self.CM)**2)
        return (p_obs-p_exp)/(1.-p_exp)

    def print_readable(self, f):
        for idx,item in enumerate(self.preds):
            f.write(u'%s\t%s\t%s\n' % (self.data[idx], self.inverse_labeldict[self.targets[idx]],
                self.inverse_labeldict[item]))

    def scores(self):
        prec =  np.diagonal(self.CM/np.sum(self.CM,axis=0)[np.newaxis,:])
        recall = np.diagonal(self.CM/np.sum(self.CM,axis=1)[:,np.newaxis])
        fscore = 2*prec*recall/(prec+recall)
        return prec, recall, fscore

def main(result_path, dict_path):
    R = Result.fromfile(result_path, dict_path)
    print("Macro-averaged F-score = {}".format(R.macroF))
    print 'Class\tPrecision\tRecall\tF-score'
    for k,v in R.inverse_labeldict.iteritems():
        print '%s\t%.2f\t%.2f\t%.2f' % (v,R.precision[k],R.recall[k],R.fscore[k])

    f = io.open('%s/readable.txt'%result_path,'w')
    R.print_readable(f)
    f.close()

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
