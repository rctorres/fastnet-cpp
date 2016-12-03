#! /usr/bin/env python

import fastnet
import numpy as np

net = fastnet.PyNeuralNetwork([3, 2, 1], ['tansig', 'tansig'], [True, True])

weights = [np.array([[1.,2.,3.],[4.,5.,6.]]), np.array([[7., 8.]])]
bias = [np.array([0.1, 0.2]), np.array([0.3])]
net.readWeights(weights, bias)

print (net.getNumLayers())
vec = np.array([1.,2.,3.,4.,5.])
ret = net.propagateInput(vec)

print(ret)