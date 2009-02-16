#ifndef MTHELPER_H
#define MTHELPER_H

#include <vector>

namespace MT
{
  pthread_cond_t trnRequest = PTHREAD_COND_INITIALIZER;
  pthread_cond_t valRequest = PTHREAD_COND_INITIALIZER;
  pthread_cond_t doneProc[2] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
  pthread_cond_t doneValProc[2] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
  pthread_mutex_t trnMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t valMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t dataMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t doneMutex[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
  pthread_mutex_t doneValMutex[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};


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
    bool stop;
  };

  void *MTValNetwork(void *param)
  {
    const REAL *output;
    ThreadParams *par = static_cast<ThreadParams*>(param);
    const unsigned initPos = par->id * par->inputSize;
    const unsigned incVal = (par->id + 1) * par->inputSize;
    while (true)
    {
      //Waiting for the signal to start.
      pthread_mutex_lock(&valMutex);
      pthread_cond_wait(&valRequest, &valMutex);
      if (par->stop)
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
      pthread_cond_signal(&doneValProc[par->id]);
      pthread_mutex_unlock(&valMutex);
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
      //Waiting for the signal to start.
      pthread_mutex_lock(&trnMutex);
      pthread_cond_wait(&trnRequest, &trnMutex);
      if (par->stop)
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
      pthread_cond_signal(&doneProc[par->id]);
      pthread_mutex_unlock(&trnMutex);
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
      thp[i].stop = false; // flag to stop the thread.
      
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
    DEBUG1("Creating MTHelper object for " << nThreads << "threads.");

    //Setting threads for being joinable.
    pthread_attr_init(&threadAttr);
    pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_JOINABLE);
/*
    doneMutex = new pthread_mutex_t [nThreads];
    doneProc = new pthread_cond_t [nThreads];
    for (unsigned i=0; i<nThreads; i++)
    {
      pthread_mutex_init(&doneMutex[i], NULL);
      pthread_cond_init(&doneProc[i], NULL);
    }
*/
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
    pthread_attr_destroy(&threadAttr);
    
    //Ending the threads.
    for (unsigned i=0; i<nThreads; i++) trnThPar[i].stop = valThPar[i].stop = true;
    pthread_cond_broadcast(&trnRequest);
    pthread_cond_broadcast(&valRequest);
    for (unsigned i=0; i<nThreads; i++) 
    {
      void *ret;
      pthread_join(trnThreads[i], &ret);
      pthread_join(valThreads[i], &ret);
      if (i) delete netVec[i];
      pthread_mutex_destroy(&doneMutex[i]);
      pthread_cond_destroy(&doneProc[i]);
      pthread_mutex_destroy(&doneValMutex[i]);
      pthread_cond_destroy(&doneValProc[i]);
    }

    pthread_mutex_destroy(&trnMutex);
    pthread_mutex_destroy(&valMutex);
    pthread_mutex_destroy(&dataMutex);
    pthread_cond_destroy(&trnRequest);
    pthread_cond_destroy(&valRequest);

    delete trnThreads;
    delete valThreads;
    delete trnThPar;
    delete valThPar;
    delete netVec;
//    delete doneMutex;
//    delete doneProc;
  };

  REAL valNetwork()
  {
    REAL gbError = 0.;
    pthread_cond_broadcast(&valRequest);
    for (unsigned i=0; i<nThreads; i++)
    {
      void *ret;
      DEBUG2("Waiting for validating thread " << i << " to finish...");
      pthread_mutex_lock(&doneValMutex[i]);
      pthread_cond_wait(&doneValProc[i], &doneValMutex[i]);
      gbError += valThPar[i].error;
      pthread_mutex_unlock(&doneValMutex[i]);
    }
    return (gbError / static_cast<REAL>(valThPar[0].numEvents));
  };


  REAL trainNetwork()
  {
    Backpropagation *mainNet = netVec[0];
    REAL gbError = 0.;
    
    //First we make all the networks having the same training status.
    for (unsigned i=1; i<nThreads; i++) (*netVec[i]) = (*mainNet);
    
    pthread_cond_broadcast(&trnRequest);
    for (unsigned i=0; i<nThreads; i++)
    {
      void *ret;
      DEBUG2("Waiting for training thread " << i << " to finish...");
      pthread_mutex_lock(&doneMutex[i]);
      pthread_cond_wait(&doneProc[i], &doneMutex[i]);
      gbError += trnThPar[i].error;
      if (i) mainNet->addToGradient(*netVec[i]);
      pthread_mutex_unlock(&doneMutex[i]);
    }
    return (gbError / static_cast<REAL>(trnThPar[0].numEvents));
  }
};
};

#endif
