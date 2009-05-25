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
  unsigned inputSize;
  unsigned outputSize;
  int numValEvents;
  DataManager *dmTrn;

public:
  StandardTraining(FastNet::Backpropagation *net, const mxArray *inTrn, const mxArray *outTrn, const mxArray *inVal, const mxArray *outVal, const unsigned bSize);

  ~StandardTraining();

  /// Applies the validating set for the network's validation.
  /**
  This method takes the one or more validating events (input and targets) and presents them
  to the network. At the end, the mean training error is returned. Since it is a validating function,
  the network is not modified, and no updating weights values are calculated. This method only
  presents the validating sets and calculates the mean validating error obtained.
  of this class are not modified inside this method, since it is only a network validating process.
  @return The mean validating error obtained after the entire training set is presented to the network.
  */
  virtual REAL valNetwork();


  /// Applies the training set for the network's training.
  /**
  This method takes the one or more training events (input and targets) and presents them
  to the network, calculating the new mean (if batch training is being used) update values 
  after each input-output pair is presented. At the end, the mean training error is returned.
  @return The mean training error obtained after the entire training set is presented to the network.
  */
  virtual REAL trainNetwork();
  
  virtual void checkSizeMismatch() const;
  
  virtual void showInfo(const unsigned nEpochs) const;
};

#endif
