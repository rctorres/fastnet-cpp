#ifndef STANDARD_H
#define STANDARD_H

#include "fastnet/matlab/Training.hxx"

using namespace FastNet;

class StandardTraining : public Training
{
private:
  const REAL *inTrnData;
  const REAL *outTrnData;
  const REAL *inValData;
  const REAL *outValData;
  unsigned numTrnEvents;
  unsigned numValEvents;
  unsigned inputSize;
  unsigned outputSize;

public:
  StandardTraining(const mxArray *inTrn, const mxArray *outTrn, const mxArray *inVal, const mxArray *outVal) : Training()
  {
    DEBUG2("Creating StandardTraining object.");
    
    if ( mxGetM(inTrn) != mxGetM(inVal) ) throw "Input training and validating events dimension does not match!";
    if ( mxGetM(outTrn) != mxGetM(outVal) ) throw "Output training and validating events dimension does not match!";
    if ( mxGetN(inTrn) != mxGetN(outTrn) ) throw "Number of input and target training events does not match!";
    if ( mxGetN(inVal) != mxGetN(outVal) ) throw "Number of input and target validating events does not match!";

    inTrnData = static_cast<REAL*>(mxGetData(inTrn));
    outTrnData = static_cast<REAL*>(mxGetData(outTrn));
    inValData = static_cast<REAL*>(mxGetData(inVal));
    outValData = static_cast<REAL*>(mxGetData(outVal));
    numTrnEvents = static_cast<unsigned>(mxGetN(inTrn));
    numValEvents = static_cast<unsigned>(mxGetN(inVal));
    inputSize = static_cast<unsigned>(mxGetM(inTrn));
    outputSize = static_cast<unsigned>(mxGetM(outTrn));
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
    REAL gbError = 0.;
    const REAL *out;

    const REAL *input = inValData;
    const REAL *target = outValData;
    for (unsigned i=0; i<numValEvents; i++)
    {
      gbError += net->applySupervisedInput(input, target, out);
      input += inputSize;
      target += outputSize;
    }
    return (gbError / static_cast<REAL>(numValEvents));
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
    unsigned evIndex;
    REAL gbError = 0.;
    const REAL *output;

    const REAL *input = inTrnData;
    const REAL *target = outTrnData;
    for (unsigned i=0; i<numTrnEvents; i++)
    {
      gbError += net->applySupervisedInput(input, target, output);
      net->calculateNewWeights(output, target);
      input += inputSize;
      target += outputSize;
    }
    return (gbError / static_cast<REAL>(numTrnEvents));
  }
  
  void checkSizeMismatch(const Backpropagation *net) const
  {
    if (inputSize != (*net)[0])
      throw "Input training or validating data do not match the network input layer size!";

    if ( outputSize != (*net)[net->getNumLayers()-1] )
      throw "Output training or validating data do not match the network output layer size!";
  };
  
  void showInfo(const unsigned nEpochs) const
  {
    REPORT("TRAINING DATA INFORMATION (Standard Network)");
    REPORT("Number of Epochs                    : " << nEpochs);
    REPORT("Total number of training events     : " << numTrnEvents);
    REPORT("Total number of validating events      : " << numValEvents);
  };
};

#endif
