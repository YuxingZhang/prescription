import pickle as pk
f = open("plot.pkl")
visualize = pk.load(f)
import pylab as Plot
Y = visualize
import numpy as np
labels = np.loadtxt("labels.txt")
Plot.scatter(Y[:,0], Y[:,1], 20, labels);
Plot.show()
