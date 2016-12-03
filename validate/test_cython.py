#! /usr/bin/env python

import fastnet
import numpy as np

net = fastnet.PyNeuralNetwork([5, 2, 1], ['tansig', 'tansig'], [True, True])
print (net.getNumLayers())
vec = np.array([1.,2.,3.,4.,5.])
ret = net.propagateInput(vec)

print(ret)