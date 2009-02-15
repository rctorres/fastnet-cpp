#ifndef STANDARD_H
#define STANDARD_H

#include "fastnet/matlab/Training.h"

using namespace FastNet;

class StandardTraining : public Training
{
private:
  MatEvents *inTrnData;
  MatEvents *outTrnData;
  MatEvents *inValData;
  MatEvents *outValData;
  unsigned trnEpochSize;

public:
  StandardTraining(const mxArray *inTrn, const mxArray *outTrn, const mxArray *inVal, const mxArray *outVal) : Training()
  {
    DEBUG2("Creating StandardTraining object.");
    inTrnData = new MatEvents (inTrn);
    outTrnData = new MatEvents (outTrn);
    inValData = new MatEvents (inVal);
    outValData = new MatEvents (outVal);
    DEBUG2("User defined epoch size? " << (epochSize != NULL));
    trnEpochSize = inTrnData->getNumEvents();
    DEBUG2("Training epoch size: " << trnEpochSize);
  };


  virtual ~StandardTraining()
  {
    delete inTrnData;
    delete outTrnData;
    delete inValData;
    delete outValData;
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
  
    for (unsigned i=0; i<inValData->getNumEvents(); i++)
    {
      // Getting the next input and target pair.
      const REAL *input = inValData->readEvent(i);
      const REAL *target = outValData->readEvent(i);
      gbError += net->applySupervisedInput(input, target, out);
    }

    return (gbError / static_cast<REAL>(inValData->getNumEvents()));
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

    for (unsigned i=0; i<trnEpochSize; i++)
    {
      // Getting the next input and target pair.
      const REAL *input = inTrnData->readRandomEvent(evIndex);
      const REAL *target = outTrnData->readEvent(evIndex);
      gbError += net->applySupervisedInput(input, target, output);

      //Calculating the weight and bias update values
      net->calculateNewWeights(output, target);
    }
    return (gbError / static_cast<REAL>(trnEpochSize));
  }
  
  vector<unsigned> getEpochSize() const
  {
    vector<unsigned> ret(1, trnEpochSize);
    return ret;
  };
  
  void checkSizeMismatch(const Backpropagation *net) const
  {
    if ( (inTrnData->getEventSize() != (*net)[0]) || (inValData->getEventSize() != (*net)[0]) )
      throw "Input training or validating data do not match the network input layer size!";

    if ( (outTrnData->getEventSize() != (*net)[net->getNumLayers()-1]) || (outValData->getEventSize() != (*net)[net->getNumLayers()-1]) )
      throw "Output training or validating data do not match the network output layer size!";
  };
  
  void showInfo(const unsigned nEpochs) const
  {
    REPORT("TRAINING DATA INFORMATION (Standard Network)");
    REPORT("Number of Epochs                    : " << nEpochs);
    REPORT("Number of training events per epoch : " << trnEpochSize);
    REPORT("Total number of training events     : " << inTrnData->getNumEvents());
    REPORT("Total number of validating events      : " << inValData->getNumEvents());
  };
};

#endif
