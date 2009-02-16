#ifndef MTHELPER_H
#define MTHELPER_H

#include <vector>

namespace MT
{
  pthread_cond_t trnRequest = PTHREAD_COND_INITIALIZER;
  pthread_cond_t valRequest = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t trnMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t valMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t dataMutex = PTHREAD_MUTEX_INITIALIZER;

  struct ThreadParams
  {
    Backpropagation *net;
    const REAL *inData;
    const REAL *outData;
    unsigned id;
    unsigned numEvents;
    unsigned inputsize;
    unsigned outputsize;
    REAL error;
  };

  void *MTValNetwork(void *param)
  {
    const REAL *output;
    ThreadParams *par = static_cast<ThreadParams*>(param);
    const unsigned initPos = par->id * par->inputsize;
    const unsigned incVal = (par->id + 1) * par->inputsize;
    while (true)
    {
      //Waiting for the signal to start.
      pthread_mutex_lock(&valMutex);
      pthread_cond_wait(&valRequest, &valMutex);
      
      par->error = 0.;
      for (unsigned i=initPos; i<par->numEvents; i+=incVal)
      {
        // Getting the next input and target pair.
        pthread_mutex_lock(&dataMutex);
        const REAL *input = par->inData[i];
        const REAL *target = par->outData[i];
        pthread_mutex_unlock(&dataMutex);
        par->error += par->net->applySupervisedInput(input, target, output);
      }
      pthread_mutex_unlock(&valMutex);
    }
    pthread_exit(NULL);
  };


  void *MTTrainNetwork(void *param)
  {
    ThreadParams *par = static_cast<ThreadParams*>(param);
    const unsigned initPos = par->id * par->inputsize;
    const unsigned incVal = (par->id + 1) * par->inputsize;
    const REAL *output;

    while (true)
    {
      //Waiting for the signal to start.
      pthread_mutex_lock(&trnMutex);
      pthread_cond_wait(&trnRequest, &trnMutex);

      par->error = 0.;
      for (unsigned i=initPos; i<par->numEvents; i+=incVal)
      {
        // Getting the next input and target pair.
        pthread_mutex_lock(&dataMutex);
        const REAL *input = par->inData[i];
        const REAL *target = par->outData[i];
        pthread_mutex_unlock(&dataMutex);
        par->error += par->net->applySupervisedInput(input, target, output);

        //Calculating the weight and bias update values
        par->net->calculateNewWeights(output, target);
      }
      pthread_mutex_unlock(&trnMutex);
    }
    pthread_exit(NULL);
  }
  


class MTHelper
{
private:
  unsigned nThreads;
  unsigned trnEpochSize;
  pthread_t *trnThreads;
  pthread_t *valThreads;
  ThreadParams *trnThPar;
  ThreadParams *valThPar;
  pthread_attr_t threadAttr;
  Backpropagation **netVec;

  DEBUG1("Starting Multi-Thread Helper Object.");
  std::vector<unsigned> distributeEvents(const unsigned totalNumEvents)
  {
    const unsigned amount = static_cast<unsigned>(totalNumEvents / nThreads);
    const unsigned rest = static_cast<unsigned>(totalNumEvents % nThreads);
    vector<unsigned> ret(nThreads, amount);
    //If we have odd division, we distribute the rest.
    for (unsigned i=0; i<rest; i++) ret[i]++;
  
#ifdef DEBUG
    DEBUG2("Total number of events: " << totalNumEvents);
    DEBUG2("Total number of threads: " << nThreads);
    for (unsigned i=0; i<nThreads; i++) DEBUG2("Number of events in thread " << i << ": " << ret[i]);
#endif

    return ret;
  }
  
  void createThreads(const REAL *inData, const REAL *outData, const unsigned numEvents, const unsigned inputSize, const unsigned outputSize, pthread_t *&th, ThreadParams *&thp, void *(*funcPtr)(void*))
  {
    th = new pthread_t[nThreads];
    thp = new ThreadParams[nThreads];
    
    DEBUG2("Setting the parameters for each thread.");
    for (unsigned i=0; i<nThreads; i++)
    {
      thp[i].id = i;
      thp[i].net = netVec[i];
      thp[i].inData = inData;
      thp[i].outData = outData;
      thp[i].numEvents = numEvents;
      thp[i].inputSize = inputSize;
      thp[i].outputSize = outputSize;
      
      const int rc = pthread_create(&th[i], &threadAttr, funcPtr, static_cast<void*>(&thp[i]));
      if (rc) throw "Impossible to create thread! Aborting...";
    }
  }



public:
  MTHelper(REAL *inTrn, REAL *outTrn, const unsigned numTrnEvents, REAL *inVal, REAL *outVal, const unsigned numValEvents,, const unsigned inputSize, const unsigned outputSize, const unsigned numThreads = 2)
  {
    nThreads = numThreads;
    DEBUG1("Creating MTHelper object for " << nThreads << "threads.");

    //Setting threads for being joinable.
    pthread_attr_init(&threadAttr);
    pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_JOINABLE);    

    //Generating copies of the neural network to be used.
    netVec = new Backpropagation* [nThreads];
    netVec[0] = net;
    for (unsigned i=1; i<nThreads; i++) netVec[i] = dynamic_cast<Backpropagation*>(net->clone());

    DEBUG1("Creating training threads.");
    createThreads(inTrn, outTrn, numTrnEvents, inputSize, outputSize, trnThPar, &MTTrainNetwork);
    DEBUG1("Creating validating threads.");
    createThreads(inVal, outVal, numValEvents, inputSize, outputSize, valThPar, &MTValNetwork);
  };

  virtual ~MTHelper()
  {
    pthread_attr_destroy(&threadAttr);
    delete trnThreads;
    delete valThreads;
    delete trnThPar;
    delete valThPar;
    for (unsigned i=1; i<nThreads; i++) delete netVec[i];
    delete netVec;
  };

  REAL valNetwork()
  {
    REAL gbError = 0.;
    pthread_cond_broadcast(&valRequest);
    for (unsigned i=0; i<nThreads; i++)
    {
      void *ret;
      pthread_join(valThreads[i], &ret);
      gbError += valThPar[i].error;
    }
    return (gbError / static_cast<REAL>(valThPar[0].numEvents));
  };


  REAL trainNetwork()
  {
    Backpropagation *mainNet = netVec[0];
    REAL gbError = 0.;
    
    //First we make all the networks having the same trining status.
    for (unsigned i=1; i<nThreads; i++) (*netVec[i]) = (*mainNet);
    
    pthread_cond_broadcast(&trnRequest);
    for (unsigned i=0; i<nThreads; i++)
    {
      void *ret;
      pthread_join(trnThreads[i], &ret);
      gbError += trnThPar[i].error;
      if (i) mainNet->addToGradient(*netVec[i]);
    }
    return (gbError / static_cast<REAL>(trnThPar[0].numEvents));
  }
  
};

};

#endif
