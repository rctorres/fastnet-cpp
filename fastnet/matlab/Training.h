#ifndef TRAINING_H
#define TRAINING_H

#include <mex.h>

#include "fastnet/neuralnet/backpropagation.h"
#include "fastnet/reporter/Reporter.h"
#include "fastnet/defines.h"

using namespace FastNet;


//This struct will hold the training info to be ruterned to the user.
struct TrainData
{
  REAL epoch;
  REAL trnError;
  REAL tstError;
};


class Training
{
protected:
  list<TrainData> trnEvolution;
  REAL bestGoal;

public:

  Training()
  {
    bestGoal = 10000000000.;
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
  
  virtual void checkSizeMismatch(const Backpropagation *net) const = 0;
  
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
    REPORT("Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << "mse (val) = " << valError);
  };
  
  virtual REAL valNetwork(Backpropagation *net) = 0;

  virtual REAL trainNetwork(Backpropagation *net) = 0;
  
  static vector<unsigned> getNumEvents(const mxArray *dataStr)
  {
    vector<unsigned> ret;
    if (mxIsCell(dataStr)) // We have multiple patterns
    {
      DEBUG2("We have a cell vector (multiple patterns)");
      for (unsigned i=0; i<mxGetN(dataStr); i++)
      {
        ret.push_back(mxGetN(mxGetCell(dataStr, i)));
        DEBUG2("Number of events for pattern " << i << ": " << ret[i]);
      }
    }
    else
    {
      DEBUG2("We have just a matrix of events.");
      ret.push_back(mxGetN(dataStr));
      DEBUG2("Number of events: " << ret[0]);
    }
    return ret;
  }
};

#endif

