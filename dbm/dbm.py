import matplotlib.pyplot as plt
import numpy as np
import math
import sys

def sigmoid(x): # sigmoid activation function
    return 1.0 / (1.0 + np.exp(-x))

class DBM:
    ''' class of deep Boltzmann machine '''

    ''' Constructor, random initialization '''
    def __init__(self, n_vis, n_hidden_1, n_hidden_2, train_x, valid_x, lr, k, K, mf_round):
        self.n_vis = n_vis                                                  # number of units in the visible layer
        self.n_hidden_1 = n_hidden_1                                        # number of units in the hidden layer 1
        self.n_hidden_2 = n_hidden_2                                        # number of units in the hidden layer 2
        self.train_x = train_x
        self.valid_x = valid_x
        self.lr = lr
        self.k = k
        self.K = K # the number of persistent chains
        self.mf_round = mf_round # number of rounds to run mean field
        self.c = np.zeros(self.n_vis)                                     
        self.b_1 = np.zeros(self.n_hidden_1)                                    
        self.b_2 = np.zeros(self.n_hidden_2)                                    
        self.W_1 = np.random.normal(0.0, 0.1, (self.n_vis, self.n_hidden_1))    # weight matrix, random Gaussian parameters, W_jk is j(hidden)-k(visible)
        self.W_2 = np.random.normal(0.0, 0.1, (self.n_hidden_1, self.n_hidden_2))    # weight matrix, random Gaussian parameters, W_jk is j(hidden)-k(visible)
        # initialize persistent chains
        self.v_sample = np.random.randint(2, size=(self.K, self.n_vis)) # samples for visible layers
        self.h_sample_1 = np.random.randint(2, size=(self.K, self.n_hidden_1)) # samples for the first hidden layers
        self.h_sample_2 = np.random.randint(2, size=(self.K, self.n_hidden_2)) # samples for the second hidden layers

    def update(self, v_batch):
        # number of data points in this batch
        N = v_batch.shape[0]

        # Compute mean field approximation of mu
        mu_1 = np.random.rand(N, self.n_hidden_1)
        mu_2 = np.random.rand(N, self.n_hidden_2)
        for n in range(N):
            for i in range(self.mf_round): # number of rounds to run mean field
                mu_1[n] = sigmoid(np.dot(self.W_1.transpose(), v_batch[n]) + np.dot(self.W_2, mu_2[n]) + self.b_1)
                mu_2[n] = sigmoid(np.dot(self.W_2.transpose(), mu_1[n]) + self.b_2)

        # sample k times in the persistent contrastive divergence
        self.sample_persistent_chain(self.k)

        # compute the update rules for W_1, W_2
        self.W_1 += self.lr * (np.dot(v_batch.transpose(), mu_1) / float(N) - np.dot(self.v_sample.transpose(), self.h_sample_1) / float(K))
        self.W_2 += self.lr * (np.dot(mu_2.transpose(), mu_1) / float(N) - np.dot(self.h_sample_1.transpose(), self.h_sample_2) / float(K))

        # compute the update rules for bias terms
        self.c   += self.lr * (np.mean(v_batch, axis=0) - np.mean(self.v_sample,   axis=0))
        self.b_1 += self.lr * (np.mean(mu_1,    axis=0) - np.mean(self.h_sample_1, axis=0))
        self.b_2 += self.lr * (np.mean(mu_2,    axis=0) - np.mean(self.h_sample_2, axis=0))

    ''' Sample the persistent chain '''
    def sample_persistent_chain(self, step):
        for i in range(self.K):
            self.v_sample[i], self.h_sample_1[i], self.h_sample_2[i], pr_x = self.sample(self.v_sample[i], self.h_sample_1[i], self.h_sample_2[i], step)

    ''' Sample the given x, h_1, h_2 for a couple of rounds, returning the probability of getting x '''
    def sample(self, x, h_1, h_2, step):
        for i in range(step):
            # sample h_1 from x and h_2
            h_1 = np.zeros(self.n_hidden_1)
            pr_h_1 = sigmoid(self.b_1 + np.dot(self.W_1.transpose(), x) + np.dot(self.W_2, h_2))
            rand_1 = np.random.rand(self.n_hidden_1)
            h_1[rand_1 <= pr_h_1] = 1.0
            h_1[rand_1 > pr_h_1] = 0.0

            # sample h_2 from h_1
            h_2 = np.zeros(self.n_hidden_2)
            pr_h_2 = sigmoid(self.b_2 + np.dot(self.W_2.transpose(), h_1))
            rand_2 = np.random.rand(self.n_hidden_2)
            h_2[rand_2 <= pr_h_2] = 1.0
            h_2[rand_2 > pr_h_2] = 0.0

            # sample x from h_1
            x = np.zeros(self.n_vis)
            pr_x = sigmoid(self.c + np.dot(self.W_1, h_1))
            rand_x = np.random.rand(self.n_vis)
            x[rand_x <= pr_x] = 1.0
            x[rand_x > pr_x] = 0.0
        return x, h_1, h_2, pr_x

    ''' Perform one epoch of training, iterating all the data in train_x '''
    def train(self):
        train_error = []
        valid_error = []
        batch_size = 100
        for i in range(self.train_x.shape[0] / batch_size): # mini batch
            self.update(self.train_x[i * batch_size: (i + 1) * batch_size - 1, :])
        train_error += [self.cross_entropy(self.train_x)]
        valid_error += [self.cross_entropy(self.valid_x)]
        print "Train error = {}, validation error = {}".format(train_error[-1], valid_error[-1])

        return train_error, valid_error

    def cross_entropy(self, x):               # TODO compute the average cross entropy error
        error_sum = 0.0
        for n in range(x.shape[0]):
            x_cur = x[n, :]
            h_1 = np.random.randint(2, size=(self.n_hidden_1, ))
            h_2 = np.random.randint(2, size=(self.n_hidden_2, ))
            xs, h_1, h_2, p_x = self.sample(np.copy(x_cur), h_1, h_2, 1)
            error_sum -= (np.dot(x_cur, np.log(p_x)) + np.dot(1.0 - x_cur, np.log(1.0 - p_x)))
        return error_sum / float(x.shape[0])

if __name__ == "__main__":
    cur_run = str(sys.argv[1]);
    print cur_run

    np.set_printoptions(threshold=np.inf)
    train_set = np.loadtxt('digitstrain.txt', delimiter=',')
    valid_set = np.loadtxt('digitsvalid.txt', delimiter=',')

    train_x = train_set[:, :-1]
    train_x[train_x <= 0.5] = 0.0
    train_x[train_x > 0.5] = 1.0
    np.random.shuffle(train_x)
    valid_x = valid_set[:, :-1]
    valid_x[valid_x <= 0.5] = 0.0
    valid_x[valid_x > 0.5] = 1.0

    n = train_x.shape[0]
    n_vis = train_x.shape[1]
    n_hidden_1 = 400
    n_hidden_2 = 400
    lr = 0.01
    k = 1 # number of steps in contrastive divergence
    K = 100 # number of chains
    mf_round = 10
    T = 1000

    rbm = DBM(n_vis, n_hidden_1, n_hidden_2, train_x, valid_x, lr, k, K, mf_round)
    
    train_error = []
    valid_error = []
    for i in range(T):
        print "Epoch {}:".format(i)
        train_e, valid_e = rbm.train()
        train_error += train_e
        valid_error += valid_e

    gibbs_step = 1000
    n_samples = 100
    x_samples = np.zeros((n_samples, n_vis))
    for i in range(n_samples):
        xs, h_1, h_2, pr_x = rbm.sample(np.random.randint(2, size=n_vis), np.random.randint(2, size=n_hidden_1), np.random.randint(2, size=n_hidden_2), gibbs_step)
        x_samples[i] = xs

    np.savetxt('x_samples_{}.csv'.format(cur_run), x_samples, delimiter=',')
    

    np.savetxt('train_error_{}.csv'.format(cur_run), np.array(train_error), delimiter=',')
    np.savetxt('valid_error_{}.csv'.format(cur_run), np.array(valid_error), delimiter=',')

    # write error rate to file
    np.savetxt('W_1_{}.csv'.format(cur_run), rbm.W_1, delimiter=',')

    quit()

    # cross entropy
    plt.figure(0)
    plt.plot(np.arange(T), train_error)
    plt.plot(np.arange(T), valid_error)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('cross_entropy_{}.png'.format(cur_run))

