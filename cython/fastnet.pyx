# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
cimport numpy as np
import numpy as np
from cpython cimport array
import array

ctypedef double REAL

cdef extern from "fastnet/neuralnet/rprop.h" namespace "FastNet":
    cdef cppclass RProp:
        RProp(const vector[unsigned] &nnodes, const vector[string] &trfFunc, const vector[bool] &useBias, const REAL deltaMin, const REAL deltaMax, const REAL initEta, const REAL incEta, const REAL decEta) except +
        const REAL* propagateInput(const REAL *input)
        REAL getWeight(unsigned layer, unsigned node, unsigned prevNode) const
        REAL getBias(unsigned layer, unsigned node) const
        unsigned getNumLayers() const
        unsigned operator[](unsigned layer) const
        void setUsingBias(const unsigned layer, const bool val)
        void setUsingBias(const bool val)
        bool isUsingBias(const unsigned layer) const
        void readWeights(const vector[ vector[ vector[REAL] ] ] &w, const vector[ vector[REAL] ] &b)
        void setFrozen(unsigned layer, unsigned node, bool frozen)
        void setFrozen(unsigned layer, bool frozen)
        bool isFrozen(unsigned layer, unsigned node) const
        bool isFrozen(unsigned layer) const
        void defrostAll()


cdef extern from "fastnet/training/DataManager.h" namespace "FastNet":
    cdef cppclass DataManager:
        DataManager() except +
        void init(const unsigned numEvents)
        unsigned numEvents() const
        unsigned eventSize() const
        unsigned getNextEventIndex()
        const REAL* operator[](const unsigned idx) const



cdef extern from "fastnet/training/Standard.h" namespace "FastNet":
    cdef cppclass StandardTraining:
        StandardTraining(RProp *net, DataManager *inTrn, DataManager *outTrn, DataManager *inVal, DataManager *outVal, const unsigned bSize) except +
        void tstNetwork(REAL &mseTst, REAL &spTst)
        void valNetwork(REAL &mseVal, REAL &spVal)
        REAL trainNetwork()
        void showInfo(const unsigned nEpochs) const



cdef class TrainInfo:
  epoch = []
  mse = []
  
  def add(self, const unsigned epochVal, const REAL mseVal):
    self.epoch.append(epochVal)
    self.mse.append(mseVal)
    


#cdef class PythonDataManager(DataManager):
#  PythonDataManager(int x) except +




cdef class PyNeuralNetwork:
    cdef RProp *c_net     # hold a C++ instance which we're wrapping

    def __cinit__(self, nnodes = None, trfFunc = None, useBias = None):
        #I have to convert from str to bytes since Cython considers C++ strings to be bytes in Python
        self.c_net = new RProp(nnodes, [t.encode() for t in trfFunc], useBias, deltaMin = 1E-6, deltaMax = 50., initEta = 0.1, incEta = 1.10, decEta = 0.5)
    
    def propagateInput(self, np.ndarray[REAL, ndim=1] input):
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
      cdef vector[vector[vector[REAL]]] cw
      cdef vector[vector[REAL]] cb
      cdef vector[REAL] aux
      cdef vector[vector[REAL]] auxW
      
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
    
    def train(self, trnSet, valSet):
      ret = TrainInfo()
      ret.add(0, 1.1)
      ret.add(1, 2.2)
      return ret