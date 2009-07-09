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
  unsigned epoch;
  REAL mse_trn;
  REAL mse_val;
  REAL sp_val;
  REAL mse_tst;
  REAL sp_tst;
  bool is_best_mse;
  bool is_best_sp;
  unsigned num_fails_mse;
  unsigned num_fails_sp;
  bool stop_mse;
  bool stop_sp;
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
  @param[in] valError The validation error obtained in that epoch.
 */
  virtual void saveTrainInfo(const unsigned epoch, const REAL mse_trn, const REAL mse_val, 
                              const REAL sp_val, const REAL mse_tst, const REAL sp_tst, 
                              const bool is_best_mse, const bool is_best_sp, 
                              const unsigned num_fails_mse, const unsigned num_fails_sp, 
                              const bool stop_mse, const bool stop_sp)
  {
    TrainData trainData;    
    trainData.epoch = epoch;
    trainData.mse_trn = mse_trn;
    trainData.mse_val = mse_val;
    trainData.sp_val = sp_val;
    trainData.mse_tst = mse_tst;
    trainData.sp_tst = sp_tst;
    trainData.is_best_mse = is_best_mse;
    trainData.is_best_sp = is_best_sp;
    trainData.num_fails_mse = num_fails_mse;
    trainData.num_fails_sp = num_fails_sp;
    trainData.stop_mse = stop_mse;
    trainData.stop_sp = stop_sp;
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
  @param[out] valError A vector containing the validation error obtained in each epoch.
  */
  virtual mxArray *flushTrainInfo()
  {
    const unsigned size = trnEvolution.size();  
    mxArray *epoch = mxCreateNumericMatrix(1, size, mxUINT32_CLASS, mxREAL);
    mxArray *mse_trn = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);
    mxArray *mse_val = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);
    mxArray *sp_val = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);
    mxArray *mse_tst = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);
    mxArray *sp_tst = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);
    mxArray *is_best_mse = mxCreateLogicalMatrix(1, size);
    mxArray *is_best_sp = mxCreateLogicalMatrix(1, size);
    mxArray *num_fails_mse = mxCreateNumericMatrix(1, size, mxUINT32_CLASS, mxREAL);
    mxArray *num_fails_sp = mxCreateNumericMatrix(1, size, mxUINT32_CLASS, mxREAL);
    mxArray *stop_mse = mxCreateLogicalMatrix(1, size);
    mxArray *stop_sp = mxCreateLogicalMatrix(1, size);

    unsigned* epoch_ptr = static_cast<unsigned*>(mxGetData(epoch));
    REAL* mse_trn_ptr = static_cast<REAL*>(mxGetData(mse_trn));
    REAL* mse_val_ptr = static_cast<REAL*>(mxGetData(mse_val));
    REAL* sp_val_ptr = static_cast<REAL*>(mxGetData(sp_val));
    REAL* mse_tst_ptr = static_cast<REAL*>(mxGetData(mse_tst));
    REAL* sp_tst_ptr = static_cast<REAL*>(mxGetData(sp_tst));
    bool* is_best_mse_ptr = static_cast<bool*>(mxGetData(is_best_mse));
    bool* is_best_sp_ptr = static_cast<bool*>(mxGetData(is_best_sp));
    unsigned* num_fails_mse_ptr = static_cast<unsigned*>(mxGetData(num_fails_mse));
    unsigned* num_fails_sp_ptr = static_cast<unsigned*>(mxGetData(num_fails_sp));
    bool* stop_mse_ptr = static_cast<bool*>(mxGetData(stop_mse));
    bool* stop_sp_ptr = static_cast<bool*>(mxGetData(stop_sp));
  
    for (list<TrainData>::const_iterator itr = trnEvolution.begin(); itr != trnEvolution.end(); itr++)
    {
      *epoch_ptr++ = itr->epoch;
      *mse_trn_ptr++ = itr->mse_trn;
      *mse_val_ptr++ = itr->mse_val;
      *sp_val_ptr++ = itr->sp_val;
      *mse_tst_ptr++ = itr->mse_tst;
      *sp_tst_ptr++ = itr->sp_tst;
      *is_best_mse_ptr++ = itr->is_best_mse;
      *is_best_sp_ptr++ = itr->is_best_sp;
      *num_fails_mse_ptr++ = itr->num_fails_mse;
      *num_fails_sp_ptr++ = itr->num_fails_sp;
      *stop_mse_ptr++ = itr->stop_mse;
      *stop_sp_ptr++ = itr->stop_sp;
    }
    
    // Creating the Matlab structure to be returned.
    const unsigned NNAMES = 12;
    const char *NAMES[] = {"epoch", "mse_trn", "mse_val", "sp_val", "mse_tst", "sp_tst",
                            "is_best_mse", "is_best_sp", "num_fails_mse", "num_fails_sp", 
                            "stop_mse", "stop_sp"};
    mxArray *ret = mxCreateStructMatrix(1,1,NNAMES,NAMES);
    mxSetField(ret, 0, "epoch", epoch);
    mxSetField(ret, 0, "mse_trn", mse_trn);
    mxSetField(ret, 0, "mse_val", mse_val);
    mxSetField(ret, 0, "sp_val", sp_val);
    mxSetField(ret, 0, "mse_tst", mse_tst);
    mxSetField(ret, 0, "sp_tst", sp_tst);
    mxSetField(ret, 0, "is_best_mse", is_best_mse);
    mxSetField(ret, 0, "is_best_sp", is_best_sp);
    mxSetField(ret, 0, "num_fails_mse", num_fails_mse);
    mxSetField(ret, 0, "num_fails_sp", num_fails_sp);
    mxSetField(ret, 0, "stop_mse", stop_mse);
    mxSetField(ret, 0, "stop_sp", stop_sp);
    return ret;
  };
  
  virtual void checkSizeMismatch() const = 0;

  virtual void showInfo(const unsigned nEpochs) const = 0;
  
  virtual void isBestNetwork(const REAL currMSEError, const REAL currSPError, bool &isBestMSE, bool &isBestSP)
  {
    if (currMSEError < bestGoal)
    {
      bestGoal = currMSEError;
      isBestMSE = true;
    }
    else isBestMSE = false;
  };
  
  virtual void showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError)
  {
    REPORT("Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << " mse (val) = " << valError);
  };

  virtual void tstNetwork(REAL &mseTst, REAL &spTst) = 0;

  virtual void valNetwork(REAL &mseVal, REAL &spVal) = 0;
  
  virtual REAL trainNetwork() = 0;  
};

#endif

