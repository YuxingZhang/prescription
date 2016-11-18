import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #W = np.loadtxt('x_samples.csv', delimiter=',')
    W = np.loadtxt('W_1.csv', delimiter=',')
    #W = np.loadtxt('result/a/W_k1_5.csv', delimiter=',')
    fig = plt.figure()
    for i in range(W.shape[1]):
        W_i = W[:, i].reshape((28, 28))
        a = fig.add_subplot(10, 10, i + 1)
        plt.axis('off')
        plt.imshow(W_i, cmap='Greys_r')
        plt.savefig("W_1.png")

    plt.show()
