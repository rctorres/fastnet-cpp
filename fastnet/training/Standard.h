#ifndef STANDARD_H
#define STANDARD_H

#include "fastnet/training/Training.h"

class StandardTraining : public Training
{
protected:
  const REAL *inTrnData;
  const REAL *outTrnData;
  const REAL *inValData;
  const REAL *outValData;
  unsigned numTrnEvents;
  unsigned numValEvents;
  unsigned inputSize;
  unsigned outputSize;

public:
  StandardTraining(const mxArray *inTrn, const mxArray *outTrn, const mxArray *inVal, const mxArray *outVal);


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
  virtual REAL valNetwork(FastNet::Backpropagation *net);


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
  virtual REAL trainNetwork(FastNet::Backpropagation *net);
  
  virtual void checkSizeMismatch(const FastNet::Backpropagation *net) const;
  
  virtual void showInfo(const unsigned nEpochs) const;
};

#endif
