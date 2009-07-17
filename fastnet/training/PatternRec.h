#ifndef PATREC_H
#define PATREC_H

#include <vector>

#include "fastnet/training/Training.h"

class PatternRecognition : public Training
{
protected:
  const REAL **inTrnList;
  const REAL **inValList;
  const REAL **inTstList;
  const REAL **targList;
  REAL **epochValOutputs;
  REAL **epochTstOutputs;
  unsigned numPatterns;
  unsigned inputSize;
  unsigned outputSize;
  bool useSP;
  bool hasTstData;
  REAL bestGoalSP;
  std::vector<DataManager*> dmTrn;
  unsigned *numValEvents;
  unsigned *numTstEvents;


  void allocateDataset(const mxArray *dataSet, const bool forTrain, 
                        const REAL **&inList, REAL **&out, unsigned *&nEv);

  void deallocateDataset(const bool forTrain, const REAL **&inList, REAL **&out, unsigned *&nEv);
  
  void getNetworkErrors(const REAL **inList, const unsigned *nEvents, REAL **epochOutputs, REAL &mseRet, REAL &spRet);


public:

  PatternRecognition(FastNet::Backpropagation *net, const mxArray *inTrn, const mxArray *inVal, 
                      const mxArray *inTst,  const bool usingSP, const unsigned bSize);

  virtual ~PatternRecognition();

  /// Calculates the SP product.
  /**
  Calculates the SP product. This method will run through the dynamic range of the outputs,
  calculating the SP product in each lambda value. Returning, at the end, the maximum SP
  product obtained.
  @return The maximum SP value obtained.
  */
  virtual REAL sp(const unsigned *nEvents, REAL **epochOutputs);

  virtual void tstNetwork(REAL &mseTst, REAL &spTst)
  {
    DEBUG2("Starting testing process for an epoch.");
    getNetworkErrors(inTstList, numTstEvents, epochTstOutputs, mseTst, spTst);
  }


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
  virtual void valNetwork(REAL &mseVal, REAL &spVal)
  {
    DEBUG2("Starting validation process for an epoch.");
    getNetworkErrors(inValList, numValEvents, epochValOutputs, mseVal, spVal);
  }


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
  virtual REAL trainNetwork();

  virtual void showInfo(const unsigned nEpochs) const;

  virtual void isBestNetwork(const REAL currMSEError, const REAL currSPError, bool &isBestMSE, bool &isBestSP);

  virtual void showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError);
  
  virtual void showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError, const REAL tstError);
};

#endif
