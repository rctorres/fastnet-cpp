/** 
@file  feedforward.h
@brief A simple feedforward network class.
*/

 
#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include <vector>
#include <cstring>

#include <mex.h>

#include "fastnet/sys/defines.h"
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
    public:

      ///Copy constructor
      /**This constructor should be used to create a new network which is an exactly copy 
        of another network.
        @param[in] net The network that we will copy the parameters from.
      */
      FeedForward(const FeedForward &net);

      /// Constructor taking the parameters for a matlab net structure.
      /**
      This constructor should be called when the network parameters are stored in a matlab
      network structure.
      @param[in] netStr The Matlab network structure as returned by newff.
      */
      FeedForward(const mxArray *netStr);

      /// Returns a clone of the object.
      /**
      Returns a clone of the calling object. The clone is dynamically allocated,
      so it must be released with delete at the end of its use.
      @return A dynamically allocated clone of the calling object.
      */
      virtual NeuralNetwork *clone();

      /// Class destructor.
      /**
       Releases all the dynamically allocated memory used by the class.
      */
      virtual ~FeedForward();
  };
}

#endif
