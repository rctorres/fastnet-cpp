#ifndef TRAINING_H
#define TRAINING_H

#include <list>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef NO_OMP
#include <omp.h>
#endif

#include <mex.h>

#include "fastnet/neuralnet/backpropagation.h"
#include "fastnet/sys/Reporter.h"
#include "fastnet/sys/defines.h"


//This struct will hold the training info to be ruterned to the user.
struct TrainData
{
  REAL epoch;
  REAL trnError;
  REAL tstError;
};


class DataManager
{
private:
  vector<unsigned>::const_iterator pos;
  vector<unsigned> vec;
  
public:
  DataManager(const unsigned numEvents)
  {
    for (unsigned i=0; i<numEvents; i++) vec.push_back(i);
    random_shuffle(vec.begin(), vec.end());
    pos = vec.begin();
  }
  
  inline unsigned size() const
  {
    return vec.size();
  }
  
  inline unsigned get()
  {
    if (pos == vec.end())
    {
      random_shuffle(vec.begin(), vec.end());
      pos = vec.begin();
    }
    return *pos++;
  }
};


class Training
{
protected:
  std::list<TrainData> trnEvolution;
  REAL bestGoal;
  FastNet::Backpropagation *net;
  FastNet::Backpropagation **netVec;
  unsigned nThreads;
  unsigned batchSize;
  int chunkSize;

  void updateNetworks()
  {
    const FastNet::Backpropagation *mainNet = netVec[0];
    for (unsigned i=1; i<nThreads; i++) (*netVec[i]) = (*mainNet);
  }

  void updateGradients()
  {
    FastNet::Backpropagation *mainNet = netVec[0];
    for (unsigned i=1; i<nThreads; i++) mainNet->addToGradient(*netVec[i]);
  }

#ifdef NO_OMP
int omp_get_num_threads() {return 1;}
int omp_get_thread_num() {return 0;}
#endif

public:

  Training(FastNet::Backpropagation *n, const unsigned bSize)
  {
    bestGoal = 10000000000.;
    batchSize = bSize;
    
    int nt;
    #pragma omp parallel shared(nt)
    {
      #pragma omp master
      nt = omp_get_num_threads();
    }

    nThreads = static_cast<unsigned>(nt);
    chunkSize = static_cast<int>(std::ceil(static_cast<float>(batchSize) / static_cast<float>(nThreads)));
    
    netVec = new FastNet::Backpropagation* [nThreads];
    net = netVec[0] = n;
    for (unsigned i=1; i<nThreads; i++) netVec[i] = new FastNet::Backpropagation(*n);
  };


  ~Training()
  {
    for (unsigned i=1; i<nThreads; i++) delete netVec[i];
    delete netVec;
  };


 /// Writes the training information of a network in a linked list.
 /**
  This method writes in a linked list in memory the information generated
  by the network during training, for improved speed. To actually stores this
  values for posterior use in matlab, you must call, at the end of the training process,
  the flushErrors method. 
  @param[in] epoch The epoch number.
  @param[in] trnError The training error obtained in that epoch.
  @param[in] tstError The testing error obtained in that epoch.
 */
  virtual void saveTrainInfo(unsigned epoch, REAL trnError, REAL tstError)
  {
    TrainData trainData;
    trainData.epoch = (REAL) epoch;
    trainData.trnError = trnError;
    trainData.tstError = tstError;
    trnEvolution.push_back(trainData);
  };

  /// Flush trining evolution info to Matlab vectors.
  /**
  Since this class, in order to optimize speed, saves the
  training information (epochs and errors) values into memory, at the end, if the user wants
  to save the final values, this method must be called. It will
  save these values stored in the linked list in Matlab vectors.
  @param[out] epoch A vector containing the epochs values.
  @param[out] trnError A vector containing the training error obtained in each epoch.
  @param[out] tstErrorA vector containing the testing error obtained in each epoch.
  */
  virtual void flushTrainInfo(mxArray *&epoch, mxArray *&trnError, mxArray *&tstError)
  {
    const unsigned size = trnEvolution.size();  
    epoch = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);
    trnError = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);
    tstError = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);

    REAL *ep = static_cast<REAL*>(mxGetData(epoch));
    REAL *trn = static_cast<REAL*>(mxGetData(trnError));
    REAL *tst = static_cast<REAL*>(mxGetData(tstError));
  
    for (list<TrainData>::const_iterator itr = trnEvolution.begin(); itr != trnEvolution.end(); itr++)
    {
      *ep++ = itr->epoch;
      *trn++ = itr->trnError;
      *tst++ = itr->tstError;
    }
  };
  
  virtual void checkSizeMismatch() const = 0;

  virtual void showInfo(const unsigned nEpochs) const = 0;
  
  virtual bool isBestNetwork(const REAL currError)
  {
    if (currError < bestGoal)
    {
      bestGoal = currError;
      return true;
    }
    return false;
  };
  
  virtual void showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError)
  {
    REPORT("Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << " mse (val) = " << valError);
  };
  
  virtual REAL valNetwork() = 0;

  virtual REAL trainNetwork() = 0;  
};

#endif

