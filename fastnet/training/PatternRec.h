#ifndef PATREC_H
#define PATREC_H

#include "fastnet/training/Training.hxx"

using namespace FastNet;

class PatternRecognition : public Training
{
protected:
  const REAL **inTrnList;
  const REAL **inValList;
  const REAL **targList;
  REAL **epochValOutputs;
  unsigned *numTrnEvents;
  unsigned *numValEvents;
  unsigned numPatterns;
  unsigned inputSize;
  unsigned outputSize;
  bool useSP;


public:

  PatternRecognition(const mxArray *inTrn, const mxArray *inVal, const bool usingSP);

  virtual ~PatternRecognition();

  /// Calculates the SP product.
  /**
  Calculates the SP product. This method will run through the dynamic range of the outputs,
  calculating the SP product in each lambda value. Returning, at the end, the maximum SP
  product obtained.
  @return The maximum SP value obtained.
  */
  virtual REAL sp();

  /// Applies the validating set of each pattern for the network's validation.
  /**
  This method takes the one or more pattern's validating events (input and targets) and presents them
  to the network. At the end, the mean training error is returned. Since it is a validating function,
  the network is not modified, and no updating weights values are calculated. This method only
  presents the validating sets and calculates the mean validating error obtained.
  @param[in] net the network class that the events will be presented to. The internal parameters
  of this class are not modified inside this method, since it is only a network validating process.
  @return The mean validating error obtained after the entire training set is presented to the network.
  */
  virtual REAL valNetwork(Backpropagation *net);


  /// Applies the training set of each pattern for the network's training.
  /**
  This method takes the one or more patterns training events (input and targets) and presents them
  to the network, calculating the new mean (if batch training is being used) update values 
  after each input-output pair of each individual pattern is presented. At the end, the mean training error is returned.
  @param[in] net the network class that the events will be presented to. At the end,
  this class is modificated, as it will contain the mean values of \f$\Delta w\f$ and \f$\Delta b\f$ obtained
  after the entire training set has been presented, but the weights are not updated at the 
  end of this function. To actually update the weights, the user must call the proper
  class's method for that.
  @return The mean training error obtained after the entire training of each pattern set is presented to the network.
  */
  virtual REAL trainNetwork(Backpropagation *net);

  virtual void checkSizeMismatch(const Backpropagation *net) const;

  virtual void showInfo(const unsigned nEpochs) const;

  virtual bool isBestNetwork(const REAL currError);

  virtual void showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError);
};

#endif
