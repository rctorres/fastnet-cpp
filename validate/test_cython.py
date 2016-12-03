#! /usr/bin/env python

import fastnet


net = fastnet.PyNeuralNetwork([10, 12, 10, 20], [b'tansig', b'purelin', b'purelin'], [True, False, True])
print (net.getNumLayers())