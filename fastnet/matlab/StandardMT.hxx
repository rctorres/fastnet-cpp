#ifndef STANDARDMT_H
#define STANDARDMT_H

#include "fastnet/matlab/Standard.hxx"
#include "fastnet/matlab/MTHelper.hxx"

using namespace FastNet;

class StandardTrainingMT : public StandardTraining
{
private:
  MT::MTHelper *mtObj;

public:
  StandardTrainingMT(Backpropagation *net, const mxArray *inTrn, const mxArray *outTrn, 
                      const mxArray *inVal, const mxArray *outVal, const unsigned numThreads = 2) 
                      : StandardTraining(inTrn, outTrn, inVal, outVal)
  {
    DEBUG2("Creating StandardTrainingMT object.");
    mtObj = new MT::MTHelper(net, inTrnData, outTrnData, numTrnEvents, inValData, outValData, numValEvents, inputSize, outputSize);
  };

  virtual ~StandardTrainingMT()
  {
    delete mtObj;
  };

  /// Applies the validating set for the network's validation.
  /**
  This method takes the one or more validating events (input and targets) and presents them
  to the network. At the end, the mean training error is returned. Since it is a validating function,
  the network is not modified, and no updating weights values are calculated. This method only
  presents the validating sets and calculates the mean validating error obtained.
  @param[in] net the network class that the events will be presented to. The internal parameters
  of this class are not modified inside this method, since it is only a network validating process.
  @return The mean validating error obtained after the entire training set is presented to the network.
  */
  REAL valNetwork(Backpropagation *net)
  {
    return mtObj->valNetwork();
  };


  /// Applies the training set for the network's training.
  /**
  This method takes the one or more training events (input and targets) and presents them
  to the network, calculating the new mean (if batch training is being used) update values 
  after each input-output pair is presented. At the end, the mean training error is returned.
  @param[in] net the network class that the events will be presented to. At the end,
  this class is modificated, as it will contain the mean values of \f$\Delta w\f$ and \f$\Delta b\f$ obtained
  after the entire training set has been presented, but the weights are not updated at the 
  end of this function. To actually update the weights, the user must call the proper
  class's method for that.
  @return The mean training error obtained after the entire training set is presented to the network.
  */
  REAL trainNetwork(Backpropagation *net)
  {
    return mtObj->trainNetwork();
  }  
};

#endif
