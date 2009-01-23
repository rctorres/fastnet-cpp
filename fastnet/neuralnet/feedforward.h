/** 
@file  feedforward.h
@brief A simple feedforward network class.
*/

 
#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include <vector>
#include <cstring>

#include <mex.h>

#include "fastnet/defines.h"
#include "fastnet/neuralnet/neuralnetwork.h"

using namespace std;


namespace FastNet
{
  /** 
  @brief    A simple feedforward class.
  @author    Rodrigo Coura Torres (torres@lps.ufrj.br)
  @version  1.0
  @date    23/01/2009

  This class should be used for network production, when no training is necessary,
  just feedforward the incoming events, fot output collection.
  */
  class FeedForward : public NeuralNetwork 
  {
    protected:
      void retropropagateError(const REAL *output, const REAL *target){};

    public:
      void calculateNewWeights(const REAL *output, const REAL *target){};
      void calculateNewWeights(const REAL *output, const REAL *target, unsigned nEv, unsigned nPat){};
      virtual void updateWeights(){};

      /// Class constructor.
      /**
       Initializes the network and also dinamically allocates and initializes to zero the 
       updating bias and weight matrices.
       @param[in] nodesDist a vector containig the number of nodes in each layer (including the input layer).
       @param[in] trfFunction a vector containig the type of transfer function in each hidden and output layer.
       @throw bad_alloc in case of error during memory allocation.
      */
      FeedForward(const vector<unsigned> &nodesDist, const vector<string> &trfFunction) 
        : NeuralNetwork(nodesDist, trfFunction){};

      ///Copy constructor
      /**This constructor should be used to create a new network which is an exactly copy 
        of another network.
        @param[in] net The network that we will copy the parameters from.
      */
      FeedForward(const FeedForward &net) : NeuralNetwork(net){};


      /// Constructor taking the parameters for a matlab net structure.
      /**
      This constructor should be called when the network parameters are stored in a matlab
      network structure.
      @param[in] netStr The Matlab network structure as returned by newff.
      */
      FeedForward(const mxArray *netStr) : NeuralNetwork(netStr){};
      

      /// Class destructor.
      /**
       Releases all the dynamically allocated memory used by the class.
      */
      virtual ~FeedForward() {};      
  };
}

#endif
