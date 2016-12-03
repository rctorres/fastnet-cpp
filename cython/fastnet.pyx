# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "fastnet/neuralnet/neuralnetwork.h" namespace "FastNet":
    cdef cppclass NeuralNetwork:
        NeuralNetwork(const vector[unsigned int] &nnodes, const vector[string] &trfFunc, const vector[bool] &useBias) except +
        unsigned int getNumLayers()

cdef class PyNeuralNetwork:
    cdef NeuralNetwork *c_net     # hold a C++ instance which we're wrapping

    def __cinit__(self, const vector[unsigned int] &nnodes, const vector[string] &trfFunc, const vector[bool] &useBias):
        self.c_net = new NeuralNetwork(nnodes, trfFunc, useBias)

    def getNumLayers(self):
      return self.c_net.getNumLayers()
