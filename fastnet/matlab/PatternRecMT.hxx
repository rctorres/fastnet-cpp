#ifndef PATRECMT_H
#define PATRECMT_H

#include "fastnet/matlab/PatternRec.hxx"
#include "fastnet/matlab/MTHelper.hxx"

class PatternRecognitionMT : public PatternRecognition
{
public:
  struct ThreadParams
  {
    FastNet::Backpropagation *net;
    const REAL **inData;
    const REAL **outData;
    unsigned id;
    unsigned numPatterns;
    unsigned *numEvents;
    unsigned inputSize;
    unsigned nThreads;
    REAL error;
    bool finishThread;
    bool threadReady;
    bool analysisReady;
  };

protected:

  unsigned nThreads;
  pthread_t *trnThreads;
  pthread_t *valThreads;
  ThreadParams *trnThPar;
  ThreadParams *valThPar;
  pthread_attr_t threadAttr;
  FastNet::Backpropagation **netVec;


  void createThreads(const REAL **inData, unsigned *numEvents, pthread_t *&th, 
                      ThreadParams *&thp, void *(*funcPtr)(void*));


public:

  PatternRecognitionMT(Backpropagation *net, const mxArray *inTrn, const mxArray *inVal, const bool usingSP, const unsigned numThreads);

  virtual ~PatternRecognitionMT();

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
  REAL valNetwork(Backpropagation *net);

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
  REAL trainNetwork(Backpropagation *net);
};

#endif
