#ifndef MTHELPER_H
#define MTHELPER_H

#include <vector>

namespace MT
{
  pthread_cond_t trnProcRequest = PTHREAD_COND_INITIALIZER;
  pthread_cond_t valProcRequest = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t trnProcMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t valProcMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t trnGetResRequest = PTHREAD_COND_INITIALIZER;
  pthread_cond_t valGetResRequest = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t trnGetResMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t valGetResMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t dataMutex = PTHREAD_MUTEX_INITIALIZER;


#ifdef DEBUG
//  pthread_mutex_t dbMutex = PTHREAD_MUTEX_INITIALIZER;
//  #define MTDB(dbmsg){pthread_mutex_lock(&dbMutex); {dbmsg}; pthread_mutex_unlock(&dbMutex);}
  #define MTDB(dbmsg)
#else
  #define MTDB(dbmsg)
#endif

  struct ThreadParams
  {
    Backpropagation *net;
    const REAL *inData;
    const REAL *outData;
    unsigned id;
    unsigned numEvents;
    unsigned inputSize;
    unsigned outputSize;
    REAL error;
    bool finishThread;
    bool threadReady;
    bool analysisReady;
  };

  inline void waitCond(bool &cond, pthread_mutex_t &mutex)
  {
    while (true)
    {
      pthread_mutex_lock(&mutex);
      if (cond)
      {
        pthread_mutex_unlock(&mutex);
        break;
      }
      pthread_mutex_unlock(&mutex);
    }
  }

  inline void safeSignal(bool &cond, pthread_mutex_t &mutex, pthread_cond_t &req)
  {
    waitCond(cond, mutex);
    pthread_cond_signal(&req);
  }

  inline void safeWait(bool &cond, pthread_mutex_t &mutex, pthread_cond_t &req)
  {
    pthread_mutex_lock(&mutex);
    cond = true;
    pthread_cond_wait(&req, &mutex);
    cond = false;
    pthread_mutex_unlock(&mutex);
  }
  
  void *MTValNetwork(void *param)
  {
    const REAL *output;
    ThreadParams *par = static_cast<ThreadParams*>(param);
    const unsigned initPos = par->id * par->inputSize;
    const unsigned incVal = (par->id + 1) * par->inputSize;
    while (true)
    {
      //Waiting for waking up...
      safeWait(par->threadReady, valProcMutex, valProcRequest);
   
      if (par->finishThread)
      {
        MTDB(DEBUG1("Exiting validating thread " << par->id));
        pthread_exit(NULL);
      }

      MTDB(DEBUG2("Starting validating process for thread " << par->id));      
      par->error = 0.;
      for (unsigned i=initPos; i<par->numEvents; i+=incVal)
      {
        // Getting the next input and target pair.
        pthread_mutex_lock(&dataMutex);
        const REAL *input = &(par->inData[i]);
        const REAL *target = &(par->outData[i]);
        pthread_mutex_unlock(&dataMutex);
        par->error += par->net->applySupervisedInput(input, target, output);
      }
      MTDB(DEBUG2("Validation process on thread " << par->id << " finished. Waiting for a new epoch..."));
      safeSignal(par->analysisReady, valGetResMutex, valGetResRequest);
    }
  };


  void *MTTrainNetwork(void *param)
  {
    ThreadParams *par = static_cast<ThreadParams*>(param);
    const unsigned initPos = par->id * par->inputSize;
    const unsigned incVal = (par->id + 1) * par->inputSize;
    const REAL *output;

    while (true)
    {
      //Waiting for waking up...
      safeWait(par->threadReady, trnProcMutex, trnProcRequest);
   
      if (par->finishThread)
      {
        MTDB(DEBUG1("Exiting training thread " << par->id));
        pthread_exit(NULL);
      }

      MTDB(DEBUG2("Starting training process for thread " << par->id));
      par->error = 0.;
      for (unsigned i=initPos; i<par->numEvents; i+=incVal)
      {
        // Getting the next input and target pair.
        pthread_mutex_lock(&dataMutex);
        const REAL *input = &(par->inData[i]);
        const REAL *target = &(par->outData[i]);
        pthread_mutex_unlock(&dataMutex);
        par->error += par->net->applySupervisedInput(input, target, output);

        //Calculating the weight and bias update values
        par->net->calculateNewWeights(output, target);
      }
      MTDB(DEBUG2("Training process on thread " << par->id << " finished. Waiting for a new epoch..."));
      safeSignal(par->analysisReady, trnGetResMutex, trnGetResRequest);
    }
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
  MTHelper(Backpropagation *net, const REAL *inTrn, const REAL *outTrn, const unsigned numTrnEvents,
            const REAL *inVal, const REAL *outVal, const unsigned numValEvents,
            const unsigned inputSize, const unsigned outputSize, const unsigned numThreads = 2)
  {
    nThreads = numThreads;
    DEBUG1("Creating MTHelper object for " << nThreads << " threads.");

    //Setting threads for being joinable.
    pthread_attr_init(&threadAttr);
    pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_JOINABLE);

    //Generating copies of the neural network to be used.
    netVec = new Backpropagation* [nThreads];
    netVec[0] = net;
    for (unsigned i=1; i<nThreads; i++) netVec[i] = dynamic_cast<Backpropagation*>(net->clone());

    DEBUG1("Creating training threads.");
    createThreads(inTrn, outTrn, numTrnEvents, inputSize, outputSize, trnThreads, trnThPar, &MTTrainNetwork);
    DEBUG1("Creating validating threads.");
    createThreads(inVal, outVal, numValEvents, inputSize, outputSize, valThreads, valThPar, &MTValNetwork);
  };

  virtual ~MTHelper()
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

    pthread_mutex_destroy(&dataMutex);
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

  REAL valNetwork()
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


  REAL trainNetwork()
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
};

#endif
