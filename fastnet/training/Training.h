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


enum ValResult {WORSE = -1, EQUAL = 0, BETTER = 1};

//This struct will hold the training info to be ruterned to the user.
struct TrainData
{
  std::vector<unsigned> epoch;
  std::vector<REAL> mse_trn;
  std::vector<REAL> mse_val;
  std::vector<REAL> sp_val;
  std::vector<ValResult> is_best_mse;
  std::vector<ValResult> is_best_sp;
  std::vector<unsigned> num_fails_mse;
  std::vector<unsigned> num_fails_sp;
  std::vector<bool> stop_mse;
  std::vector<bool> stop_sp;
  
  const unsigned size() const {return epoch.size();};
  
};


class Training
{
protected:
  TrainData trnEvolution;
  REAL bestGoal;
  FastNet::Backpropagation *mainNet;
  FastNet::Backpropagation **netVec;
  unsigned nThreads;
  unsigned batchSize;
  int chunkSize;

  void updateGradients()
  {
    for (unsigned i=1; i<nThreads; i++) mainNet->addToGradient(*netVec[i]);
  }

  virtual void updateWeights()
  {
    mainNet->updateWeights(batchSize);
    for (unsigned i=1; i<nThreads; i++) (*netVec[i]) = (*mainNet);
  };


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
    mainNet = netVec[0] = n;
    for (unsigned i=1; i<nThreads; i++) netVec[i] = new FastNet::Backpropagation(*n);
  };


  virtual ~Training()
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
                              const REAL sp_val,  
                              const ValResult is_best_mse, const ValResult is_best_sp, 
                              const unsigned num_fails_mse, const unsigned num_fails_sp, 
                              const bool stop_mse, const bool stop_sp)
  {
    trnEvolution.epoch.push_back(epoch);
    trnEvolution.mse_trn.push_back(mse_trn);
    trnEvolution.mse_val.push_back(mse_val);
    trnEvolution.sp_val.push_back(sp_val);
    trnEvolution.is_best_mse.push_back(is_best_mse);
    trnEvolution.is_best_sp.push_back(is_best_sp);
    trnEvolution.num_fails_mse.push_back(num_fails_mse);
    trnEvolution.num_fails_sp.push_back(num_fails_sp);
    trnEvolution.stop_mse.push_back(stop_mse);
    trnEvolution.stop_sp.push_back(stop_sp);
  };
  
  const TrainData& getTrainInfo() const
  {
    return trnEvolution;
  };
  
  virtual void showInfo(const unsigned nEpochs) const = 0;
  
  virtual void isBestNetwork(const REAL currMSEError, const REAL currSPError, ValResult &isBestMSE, ValResult &isBestSP)
  {
    if (currMSEError < bestGoal)
    {
      bestGoal = currMSEError;
      isBestMSE = BETTER;
    }
    else if (currMSEError > bestGoal) isBestMSE = WORSE;
    else isBestMSE = EQUAL;
  };
  
  virtual void showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError)
  {
    REPORT("Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << " mse (val) = " << valError);
  };

  virtual void valNetwork(REAL &mseVal, REAL &spVal) = 0;
  
  virtual REAL trainNetwork() = 0;  
};

#endif

