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
    NeuralNetwork *net;
    MatEvents *inData;
    MatEvents *outData;
    unsigned initEvPos;
    unsigned endEvPos;
    unsigned numEvents;
    REAL error;
  };

  void *MTValNetwork(void *par)
  {
    const REAL *out;
    ThreadParams *par = static_cast<ThreadParams*>(par);

    while (true)
    {
      //Waiting for the signal to start.
      pthread_mutex_lock(&valMutex);
      pthread_cond_wait(&valRequest, &valMutex);
      
      par->error = 0.;
      for (unsigned i=0; i<par->inData->getNumEvents(); i++)
      {
        // Getting the next input and target pair.
        const REAL *input = par->inData->readEvent(i);
        const REAL *target = par->outData->readEvent(i);
        par->error += par->net->applySupervisedInput(input, target, out);
      }

      pthread_mutex_unlock(&valMutex);
    }
    pthread_exit(NULL);
  };


  void *MTTrainNetwork(void *par)
  {
    ThreadParams *par = static_cast<ThreadParams*>(par);
    unsigned evIndex;
    const REAL *output;

    while (true)
    {
      //Waiting for the signal to start.
      pthread_mutex_lock(&trnMutex);
      pthread_cond_wait(&trnRequest, &trnMutex);

      par->error = 0.;
      for (unsigned i=0; i<trnEpochSize; i++)
      {
        // Getting the next input and target pair.
        const REAL *input = par->inData->readRandomEvent(evIndex);
        const REAL *target = par->outData->readEvent(evIndex);
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
  std::vector<NeuralNetwork*> netVec;

  
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
  
  void createThreads(MatEvents *inData, MatEvents *outData, pthread_t *th, ThreadParams *thp, void *(*funcPtr)(void*))
  {
    th = new pthread_t[nThreads];
    thp = new ThreadParams[nThreads];
    
    //Getting the number of events per thread.
    std::vector<unsigned> evThreads = distributeEvents(inData->getNumEvents(), nThreads);

    DEBUG2("Setting the parameters for each thread.");
    unsigned evStartPos = 0;
    for (unsigned i=0; i<nThreads; i++)
    {
      thp[i].net = netVec[i];
      thp[i].inData = inData;
      thp[i].outData = outData;
      thp[i].initEvPos = evStartPos;
      thp[i].endEvPos = evStartPos + evThreads[i];
      thp[i].numEvents = evThreads[i];
      evStartPos += evThreads[i];
      
      const int rc = pthread_create(&th[i], &threadAttr, funcPtr, static_cast<void*>(&thp[i]));
      if (rc) throw "Impossible to create thread! Aborting...";
    }
  }



public:
  MTHelper(NeuralNetwork *net, MatEvents *inTrn, MatEvents *outTrn, MatEvents *inVal, MatEvents *outVal, const unsigned epochSize, const unsigned numThreads = 2)
  {
    DEBUG1("Creating MTHelper object.");
    trnEpochSize = epochSize;
    DEBUG2("Training epoch size: " << trnEpochSize);
    nThreads = numThreads;

    //Setting threads for being joinable.
    pthread_attr_init(&threadAttr);
    pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_JOINABLE);
    
    //Creating copies of the neural network to be trained.
    netVec.push_back(net);
    for (unsigned i=0; i<(nThreads - 1); i++) netVec.push_back(net->clone());
    
    DEBUG1("Creating training threads.");
    createThreads(inTrn, outTrn, trnThreads, trnThPar, &MTTrainNetwork);
    DEBUG1("Creating validating threads.");
    createThreads(inVal, outVal, valThreads, valThPar, &MTValNetwork);
  };


  virtual ~StandardTrainingMT()
  {
    pthread_attr_destroy(&attr);
    delete trnThreads;
    delete valThreads;
    delete trnThPar;
    delete valThPar;
    for (vector<NeuralNetwork*>::iterator itr = (netVec.begin()+1); itr !=netVec.end(); itr++) delete *itr;
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
    return (gbError / static_cast<REAL>(valThPar[0].inData->getNumEvents()));
  };


  REAL trainNetwork()
  {
    NeuralNetwork *mainNet = netVec[0];
    REAL gbError = 0.;
    
    //First we make all the networks having the same trining status.
    for (unsigned i=1; i<nThreads; i++) (*netVec[i]) = (*mainNet);
    
    pthread_cond_broadcast(&trnRequest);
    for (unsigned i=0; i<nThreads; i++)
    {
      void *ret;
      pthread_join(trnThreads[i], &ret);
      gbError += trnThPar[i].error;
      if (i) mainNet->addToGradient(netVec[i]);
    }
    return (gbError / static_cast<REAL>(trnThPar[0].inData->getNumEvents()));
  }
  
};

};

#endif
