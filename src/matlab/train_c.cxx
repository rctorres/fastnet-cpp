/** 
@file  ntrain.cpp
@brief The Matlab's ntrain function definition file.

 This file implements the function that is called by matlab when the matlab's train function
 is called. This function reads the matlab arguments (specified in "args"),
 creates the neural network class and initializes the data sets with this arguments.
 After that the training is performed and, at the end, the results are converted to
 Octave variables to be returned by to this environment by this function.
*/

#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mex.h>

#include "fastnet/sys/Reporter.h"
#include "fastnet/neuralnet/backpropagation.h"
#include "fastnet/neuralnet/rprop.h"
#include "fastnet/training/Standard.h"
#include "fastnet/training/PatternRec.h"
#include "matlabbp.hxx"
#include "matlabrp.hxx"
#include "mxdatamanager.hxx"

using namespace std;
using namespace FastNet;

/// Index, in the arguments list, of the neural network structure.
const unsigned NET_STR_IDX = 0;

/// Index, in the arguments list, of the neural network train parameters structure.
const unsigned NET_TRN_STR_IDX = 1;

/// Index, in the arguments list, of the input training events.
const unsigned IN_TRN_IDX = 2;

/// Index, in the arguments list, of the output training events.
const unsigned OUT_TRN_IDX = 3;

/// Index, in the arguments list, of the input validating events.
const unsigned IN_VAL_IDX = 4;

/// Index, in the arguments list, of the output validating events.
const unsigned OUT_VAL_IDX = 5;

/// Index, in the return vector, of the network structure after training.
const unsigned OUT_NET_IDX = 0;

/// Index, in the return vector, of the structure containing the training evolution.
const unsigned OUT_TRN_EVO = 1;


bool isEmpty(const mxArray *mat)
{
  return ( (!mxGetM(mat)) && (!mxGetN(mat)) );
}

/// Matlab 's main function.
void mexFunction(int nargout, mxArray *ret[], int nargin, const mxArray *args[])
{
  Backpropagation *net = nullptr;
  Training *train = nullptr;
  MatlabBP *matHandler = nullptr;
  MxDataManager *inTrn = nullptr;
  MxDataManager *outTrn = nullptr;
  MxDataManager *inVal = nullptr;
  MxDataManager *outVal = nullptr;
  std::vector<DataManager*> patInTrn, patInVal;
  
  try
  {
    //If the out_trn is not empty, then is a standard training.
    const bool stdTrainingType = !isEmpty(args[OUT_TRN_IDX]);

    //Reading the configuration structure
    const mxArray *netStr = args[NET_STR_IDX];

    //Reading the showing period, epochs and max_fail.
    const mxArray *trnParam =  args[NET_TRN_STR_IDX];
    const unsigned nEpochs = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "epochs")));
    const unsigned show = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "show")));
    const unsigned fail_limit = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "max_fail")));
    const unsigned batchSize = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "batchSize")));
    const REAL signalWeight = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "sp_signal_weight")));
    const REAL noiseWeight = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "sp_noise_weight")));

    //Selecting the training type by reading the training agorithm.    
    const string trnType = mxArrayToString(mxGetField(netStr, 0, "trainFcn"));
    if (trnType == TRAINRP_ID)
    {
      matHandler = new MatlabRP(netStr, trnParam);
      if (show) REPORT("Starting Resilient Backpropagation training...");
    }
    else if (trnType == TRAINGD_ID)
    {
      matHandler = new MatlabBP(netStr, trnParam);
      if (show) REPORT("Starting Gradient Descendent training...");
    }
    else throw "Invalid training algorithm option!";

    //Getting the network class for the training.
    net = matHandler->getNetwork();

    //Getting whether we will use SP stoping criteria.
    const bool useSP = static_cast<bool>(mxGetScalar(mxGetField(trnParam, 0, "useSP")));
    
    //Creating the object for the desired training type.
    if (stdTrainingType)
    {
      inTrn = new MxDataManager(args[IN_TRN_IDX]);
      outTrn = new MxDataManager(args[OUT_TRN_IDX]);
      inVal = new MxDataManager(args[IN_VAL_IDX]);
      outVal = new MxDataManager(args[OUT_VAL_IDX]);
      train = new StandardTraining(net, inTrn, outTrn, inVal, outVal, batchSize);
    }
    else // It is a pattern recognition network.
    {
      for (auto i=0; i<mxGetN(args[IN_TRN_IDX]); i++)
      {
        patInTrn.push_back(new MxDataManager(mxGetCell(args[IN_TRN_IDX], i)));
        patInVal.push_back(new MxDataManager(mxGetCell(args[IN_VAL_IDX], i)));
      }
      train = new PatternRecognition(net, &patInTrn, &patInVal, useSP, batchSize, signalWeight, noiseWeight);
    }

#ifdef DEBUG
    //Displaying the training info before starting.
    net->showInfo();
    train->showInfo(nEpochs);
#endif
    
    if (show) REPORT("Network Training Status:");
        
    // Performing the training.
    unsigned num_fails_mse = 0;
    unsigned num_fails_sp = 0;
    unsigned dispCounter = 0;
    REAL mse_val, sp_val;
    mse_val = sp_val = 0.;
    ValResult is_best_mse, is_best_sp;
    bool stop_mse, stop_sp;

    //Calculating the max_fail limits for each case (MSE and SP, if the case).
    const unsigned fail_limit_mse = (useSP) ? (fail_limit / 2) : fail_limit;
    const unsigned fail_limit_sp = (useSP) ? fail_limit : 0;
    ValResult &is_best = (useSP) ? is_best_sp :  is_best_mse;
    REAL &val_data = (useSP) ? sp_val : mse_val;

    for (unsigned epoch=0; epoch<nEpochs; epoch++)
    {
      //Training the network and calculating the new weights.
      const REAL mse_trn = train->trainNetwork();

      //Validating the new network.
      train->valNetwork(mse_val, sp_val);

      // Saving the best weight result.
      train->isBestNetwork(mse_val, sp_val, is_best_mse, is_best_sp);
      
      if (is_best_mse == BETTER) num_fails_mse = 0;
      else if (is_best_mse == WORSE) num_fails_mse++;

      if (is_best_sp == BETTER) num_fails_sp = 0;
      else if (is_best_sp == WORSE) num_fails_sp++;
      
      if (is_best == BETTER) net->saveBestTrain();

      //Showing partial results at every "show" epochs (if show != 0).
      if (show)
      {
        if (!dispCounter)
        {
          train->showTrainingStatus(epoch, mse_trn, val_data);
        }
        dispCounter = (dispCounter + 1) % show;
      }

      //Knowing whether the criterias are telling us to stop.
      stop_mse = num_fails_mse >= fail_limit_mse;
      stop_sp = num_fails_sp >= fail_limit_sp;

      //Saving the training evolution info.
      train->saveTrainInfo(epoch, mse_trn, mse_val, sp_val, is_best_mse, 
                            is_best_sp, num_fails_mse, num_fails_sp, stop_mse, stop_sp);

      if ( (stop_mse) && (stop_sp) )
      {
        if (show) REPORT("Maximum number of failures reached. Finishing training...");
        break;
      }
    }

    // Generating a copy of the network structure passed as input.
    ret[OUT_NET_IDX] = mxDuplicateArray(netStr);
  
    //Saving the training results.
    matHandler->flushBestTrainWeights(ret[OUT_NET_IDX], net);
    
    //Returning the training evolution info.
    DEBUG1("Flushing training info.");
    ret[OUT_TRN_EVO] = train->flushTrainInfo();
    
    //Deleting the allocated memory.
    DEBUG1("Releasing all allocated memory.");
    delete matHandler;
    delete net;
    delete train;
    if (inTrn != nullptr) delete inTrn;
    if (outTrn != nullptr) delete outTrn;
    if (inVal != nullptr) delete inVal;
    if (outVal != nullptr) delete outVal;
    for (const auto &x : patInTrn) delete x;
    for (const auto &x : patInVal) delete x;
    if (show) REPORT("Training process finished!");
  }
  catch(bad_alloc xa)
  {
    FATAL("Error on allocating memory!");
    if (matHandler != nullptr) delete net;
    if (net != nullptr) delete net;
    if (train != nullptr) delete train;
    if (inTrn != nullptr) delete inTrn;
    if (outTrn != nullptr) delete outTrn;
    if (inVal != nullptr) delete inVal;
    if (outVal != nullptr) delete outVal;
    for (const auto &x : patInTrn) delete x;
    for (const auto &x : patInVal) delete x;
  }
  catch (const char *msg)
  {
    FATAL(msg);
    if (matHandler != nullptr) delete net;
    if (net != nullptr) delete net;
    if (train != nullptr) delete train;
    if (inTrn != nullptr) delete inTrn;
    if (outTrn != nullptr) delete outTrn;
    if (inVal != nullptr) delete inVal;
    if (outVal != nullptr) delete outVal;
    for (const auto &x : patInTrn) delete x;
    for (const auto &x : patInVal) delete x;
  }
  
}
