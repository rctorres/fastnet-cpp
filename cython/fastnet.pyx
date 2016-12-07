# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
cimport numpy as np
import numpy as np
from cpython cimport array
import array

cdef extern from "fastnet/neuralnet/neuralnetwork.h" namespace "FastNet":
    cdef cppclass NeuralNetwork:
        NeuralNetwork(const vector[unsigned] &nnodes, const vector[string] &trfFunc, const vector[bool] &useBias) except +
        const double* propagateInput(const double *input)
        double getWeight(unsigned layer, unsigned node, unsigned prevNode) const
        double getBias(unsigned layer, unsigned node) const
        unsigned getNumLayers() const
        unsigned operator[](unsigned layer) const
        void setUsingBias(const unsigned layer, const bool val)
        void setUsingBias(const bool val)
        bool isUsingBias(const unsigned layer) const
        void readWeights(const vector[ vector[ vector[double] ] ] &w, const vector[ vector[double] ] &b)


cdef extern from "fastnet/neuralnet/backpropagation.h" namespace "FastNet":
    cdef cppclass Backpropagation:
        Backpropagation(const vector[unsigned] &nnodes, const vector[string] &trfFunc, const vector[bool] &useBias,  const double learningRate, const double decFactor) except +
        void setFrozen(unsigned layer, unsigned node, bool frozen)
        void setFrozen(unsigned layer, bool frozen)
        bool isFrozen(unsigned layer, unsigned node) const
        bool isFrozen(unsigned layer) const
        void defrostAll()



cdef extern from "fastnet/neuralnet/rprop.h" namespace "FastNet":
    cdef cppclass RProp:
        RProp(const vector[unsigned] &nnodes, const vector[string] &trfFunc, const vector[bool] &useBias, const double deltaMin, const double deltaMax, const double initEta, const double incEta, const double decEta) except +


cdef class PyNeuralNetwork:
    cdef NeuralNetwork *c_net     # hold a C++ instance which we're wrapping

    def __cinit__(self, nnodes = None, trfFunc = None, useBias = None):
        #I have to convert from str to bytes since Cython considers C++ strings to be bytes in Python
        self.c_net = new NeuralNetwork(nnodes, [t.encode() for t in trfFunc], useBias)
    
    def propagateInput(self, np.ndarray[np.double_t, ndim=1] input):
      input = np.ascontiguousarray(input)
      c_ret = self.c_net.propagateInput(&input[0])
      numNodes = self.c_net[0][self.c_net.getNumLayers() - 1] #[0] is needed since it is a pointer.
      return np.array([c_ret[i] for i in range(numNodes)])

    def getWeight(self, layer, node, prevNode):
      return self.c_net.getWeight(layer, node, prevNode)
    
    def getBias(self, layer, node):
      return self.c_net.getBias(layer, node)

    def getNumLayers(self):
      return self.c_net.getNumLayers()

    def __getitem__(self, l):
      return self.c_net[0][l]

    def setUsingBias(self, layer, val):
      self.c_net.setUsingBias(layer, val)

    def setUsingBias(self, val):
      self.c_net.setUsingBias(val)
      
    def isUsingBias(self, layer):
      return self.c_net.isUsingBias(layer)

    def readWeights(self, weight, bias):
      nLayers = len(weight)
      cdef vector[vector[vector[double]]] cw
      cdef vector[vector[double]] cb
      cdef vector[double] aux
      cdef vector[vector[double]] auxW
      
      for w,b in zip(weight, bias):
        #Dealing with bias
        aux.clear()
        for v in b: aux.push_back(v)
        cb.push_back(aux)
        
        #Dealing with weights
        auxW.clear()
        for r in range(w.shape[0]):
          aux.clear()
          for v in w[r,:]: aux.push_back(v)
          auxW.push_back(aux)
        cw.push_back(auxW)

      self.c_net.readWeights(cw, cb)

    def sim(self, dataset):
      nEvents, inSize = dataset.shape
      netInSize = self.c_net[0][0]

      if inSize != netInSize:
        raise ValueError('Network input layer does not match the event size ({} != {})'.format(netInSize, inSize))
      
      outSize = self.c_net[0][self.c_net.getNumLayers() - 1] #[0] is needed since it is a pointer.
      ret = np.zeros((nEvents, outSize))
      for e, event in enumerate(dataset):
        ret[e,:] = self.propagateInput(event)
      return ret