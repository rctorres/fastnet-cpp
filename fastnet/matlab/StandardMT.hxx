#ifndef STANDARDMT_H
#define STANDARDMT_H

#include "fastnet/matlab/Standard.hxx"
#include "fastnet/matlab/MTHelper.hxx"

using namespace FastNet;
using namespace MT;

  void *MTValNetwork(void *param)
  {
    ThreadParams *par = static_cast<ThreadParams*>(param);
    const unsigned inputStep = par->nThreads * par->inputSize;
    const unsigned outputStep = par->nThreads * par->outputSize;
    const REAL *output;

    while (true)
    {
      //Waiting for waking up...
      safeWait(par->threadReady, valProcMutex, valProcRequest);
   
      if (par->finishThread) pthread_exit(NULL);

      par->error = 0.;
      const REAL *input = par->inData + (par->id * par->inputSize);
      const REAL *target = par->outData + (par->id * par->outputSize);
      for (unsigned i=0; i<par->numEvents; i+=par->nThreads)
      {
        par->error += par->net->applySupervisedInput(input, target, output);
        input += inputStep;
        target += outputStep;
      }
      safeSignal(par->analysisReady, valGetResMutex, valGetResRequest);
    }
  };


  void *MTTrainNetwork(void *param)
  {
    ThreadParams *par = static_cast<ThreadParams*>(param);
    const unsigned inputStep = par->nThreads * par->inputSize;
    const unsigned outputStep = par->nThreads * par->outputSize;
    const REAL *output;

    while (true)
    {
      //Waiting for waking up...
      safeWait(par->threadReady, trnProcMutex, trnProcRequest);
   
      if (par->finishThread) pthread_exit(NULL);

      par->error = 0.;
      const REAL *input = par->inData + (par->id * par->inputSize);
      const REAL *target = par->outData + (par->id * par->outputSize);
      for (unsigned i=0; i<par->numEvents; i+=par->nThreads)
      {
        par->error += par->net->applySupervisedInput(input, target, output);
        par->net->calculateNewWeights(output, target);
        input += inputStep;
        target += outputStep;
      }
      safeSignal(par->analysisReady, trnGetResMutex, trnGetResRequest);
    }
  }


class StandardTrainingMT : public StandardTraining
{
private:
  unsigned nThreads;
  pthread_t *trnThreads;
  pthread_t *valThreads;
  ThreadParams *trnThPar;
  ThreadParams *valThPar;
  pthread_attr_t threadAttr;
  Backpropagation **netVec;


  void createThreads(const REAL *inData, const REAL *outData, const unsigned numEvents, 
                      const unsigned inputSize, const unsigned outputSize, pthread_t *&th, 
                      ThreadParams *&thp, void *(*funcPtr)(void*))
  {
    DEBUG1("Starting Multi-Thread Helper Object.");
    th = new pthread_t[nThreads];
    thp = new ThreadParams[nThreads];
    
    DEBUG2("Setting the parameters for each thread.");
    for (unsigned i=0; i<nThreads; i++)
    {
      thp[i].nThreads = nThreads;
      thp[i].id = i;
      thp[i].net = netVec[i];
      thp[i].inData = inData;
      thp[i].outData = outData;
      thp[i].numEvents = numEvents;
      thp[i].inputSize = inputSize;
      thp[i].outputSize = outputSize;
      thp[i].finishThread = thp[i].threadReady = thp[i].analysisReady = false;
      
      const int rc = pthread_create(&th[i], &threadAttr, funcPtr, static_cast<void*>(&thp[i]));
      if (rc) throw "Impossible to create thread! Aborting...";
    }
  }


public:
  StandardTrainingMT(Backpropagation *net, const mxArray *inTrn, const mxArray *outTrn, 
                      const mxArray *inVal, const mxArray *outVal, const unsigned numThreads = 2) 
                      : StandardTraining(inTrn, outTrn, inVal, outVal)
  {
    DEBUG2("Creating StandardTrainingMT object.");

    nThreads = numThreads;

    //Setting threads for being joinable.
    pthread_attr_init(&threadAttr);
    pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_JOINABLE);

    //Generating copies of the neural network to be used.
    netVec = new Backpropagation* [nThreads];
    netVec[0] = net;
    for (unsigned i=1; i<nThreads; i++) netVec[i] = dynamic_cast<Backpropagation*>(net->clone());

    DEBUG1("Creating training threads.");
    createThreads(inTrnData, outTrnData, numTrnEvents, inputSize, outputSize, trnThreads, trnThPar, &MTTrainNetwork);
    DEBUG1("Creating validating threads.");
    createThreads(inValData, outValData, numValEvents, inputSize, outputSize, valThreads, valThPar, &MTValNetwork);
  };

  virtual ~StandardTrainingMT()
  {    
    //Ending the threads.
    for (unsigned i=0; i<nThreads; i++)
    {
      void *ret;
      trnThPar[i].finishThread = valThPar[i].finishThread = true;
      waitCond(trnThPar[i].threadReady, trnProcMutex);
      waitCond(valThPar[i].threadReady, valProcMutex);
      pthread_cond_broadcast(&trnProcRequest);
      pthread_cond_broadcast(&valProcRequest);    
      pthread_join(trnThreads[i], &ret);
      pthread_join(valThreads[i], &ret);
    }

    pthread_cond_destroy(&trnProcRequest);
    pthread_cond_destroy(&valProcRequest);
    pthread_mutex_destroy(&trnProcMutex);
    pthread_mutex_destroy(&valProcMutex);
    pthread_cond_destroy(&trnGetResRequest);
    pthread_cond_destroy(&valGetResRequest);
    pthread_mutex_destroy(&trnGetResMutex);
    pthread_mutex_destroy(&valGetResMutex);
    pthread_attr_destroy(&threadAttr);

    delete trnThreads;
    delete valThreads;
    delete trnThPar;
    delete valThPar;
    delete netVec;
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
    for (unsigned i=0; i<nThreads; i++) waitCond(valThPar[i].threadReady, valProcMutex);
    pthread_cond_broadcast(&valProcRequest);

    for (unsigned i=0; i<nThreads; i++)
    {
      void *ret;
      DEBUG2("Waiting for validating thread " << i << " to finish...");
      safeWait(valThPar[i].analysisReady, valGetResMutex, valGetResRequest);
      DEBUG2("Starting analysis for validating thread " << i << "...");
      gbError += valThPar[i].error;
    }
    return (gbError / static_cast<REAL>(valThPar[0].numEvents));
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
    Backpropagation *mainNet = netVec[0];
    REAL gbError = 0.;
    
    //First we make all the networks having the same training status.
    for (unsigned i=1; i<nThreads; i++) (*netVec[i]) = (*mainNet);

    for (unsigned i=0; i<nThreads; i++) waitCond(trnThPar[i].threadReady, trnProcMutex);
    pthread_cond_broadcast(&trnProcRequest);
    
    for (unsigned i=0; i<nThreads; i++)
    {
      void *ret;
      DEBUG2("Waiting for training thread " << i << " to finish...");
      safeWait(trnThPar[i].analysisReady, trnGetResMutex, trnGetResRequest);
      DEBUG2("Starting analysis for training thread " << i << "...");
      gbError += trnThPar[i].error;
      if (i) mainNet->addToGradient(*netVec[i]);
    }
    return (gbError / static_cast<REAL>(trnThPar[0].numEvents));
  }  
};

#endif