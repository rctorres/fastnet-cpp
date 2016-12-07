#! /usr/bin/env python

import fastnet
import numpy as np

net = fastnet.PyNeuralNetwork([3, 2, 1], ['tansig', 'tansig'], [True, True])

weights = [np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]]), np.array([[0.7, 0.8]])]
bias = [np.array([0.11, 0.22]), np.array([0.33])]
net.readWeights(weights, bias)

print (net.getNumLayers())
vec = np.array([1.,2.,3., 4., 5.])
ret = net.propagateInput(vec)
print(ret)

mat = np.array([[1.,2.,3.], [6.,7.,8.], [11.,12.,13.]])
ret = net.sim(mat)
print(ret)

trnInfo = net.train(None, None)

