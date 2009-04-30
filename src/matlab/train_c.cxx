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

using namespace std;
using namespace FastNet;

/// Index, in the arguments list, of the neural network structure.
const unsigned NET_STR_IDX = 0;

/// Index, in the arguments list, of the input training events.
const unsigned IN_TRN_IDX = 1;

/// Index, in the arguments list, of the output training events.
const unsigned OUT_TRN_IDX = 2;

/// Index, in the arguments list, of the input validating events.
const unsigned IN_VAL_IDX = 3;

/// Index, in the arguments list, of the output validating events.
const unsigned OUT_VAL_IDX = 4;

/// Index, in the arguments list, of the batch size.
const unsigned BATCH_SIZE_IDX = 5;

/// Index, in the return vector, of the network structure after training.
const unsigned OUT_NET_IDX = 0;

/// Index, in the return vector, of the vector containing the epoch values.
const unsigned OUT_EPOCH_IDX = 1;

/// Index, in the return vector, of the vector containing the training error obtained in each epoch.
const unsigned OUT_TRN_ERROR_IDX = 2;

//Index, in the return vector, of the vector containing the validating error obtained in each epoch.
const unsigned OUT_VAL_ERROR_IDX = 3;


bool isEmpty(const mxArray *mat)
{
  return ( (!mxGetM(mat)) && (!mxGetN(mat)) );
}


/// Matlab 's main function.
void mexFunction(int nargout, mxArray *ret[], int nargin, const mxArray *args[])
{
  Backpropagation *net = NULL;
  Training *train = NULL;
  
  try
  {
    //If the out_trn is not empty, then is a standard training.
    const bool stdTrainingType = !isEmpty(args[OUT_TRN_IDX]);

    //Reading the configuration structure
    const mxArray *netStr = args[NET_STR_IDX];

    //Reading the showing period, epochs and max_fail.
    const mxArray *trnParam =  mxGetField(netStr, 0, "trainParam");
    const unsigned nEpochs = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "epochs")));
    const unsigned show = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "show")));
    const unsigned maxFail = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "max_fail")));

    //Selecting the training type by reading the training agorithm.    
    const string trnType = mxArrayToString(mxGetField(netStr, 0, "trainFcn"));
    if (trnType == TRAINRP_ID)
    {
      net = new RProp(netStr);
      if (show) REPORT("Starting Resilient Backpropagation training...");
    }
    else if (trnType == TRAINGD_ID)
    {
      net = new Backpropagation(netStr);
      if (show) REPORT("Starting Gradient Descendent training...");
    }
    else throw "Invalid training algorithm option!";

    //Creating the object for the desired training type.
    const unsigned batchSize = static_cast<unsigned>(mxGetScalar(args[BATCH_SIZE_IDX]));
    if (stdTrainingType)
    {
      train = new StandardTraining(net, args[IN_TRN_IDX], args[OUT_TRN_IDX], args[IN_VAL_IDX], args[OUT_VAL_IDX], batchSize);
    }
    else // It is a pattern recognition network.
    {
      //Getting whether we will use SP stoping criteria.
      const mxArray *usingSP = mxGetField(mxGetField(netStr, 0, "userdata"), 0, "useSP");
      const bool useSP = static_cast<bool>(mxGetScalar(usingSP));
      train = new PatternRecognition(net, args[IN_TRN_IDX], args[IN_VAL_IDX], useSP, batchSize);
    }

    //Checking if the training and validating input data sizes match the network's input layer.
    train->checkSizeMismatch();

#ifdef DEBUG
    //Displaying the training info before starting.
    net->showInfo();
    train->showInfo(nEpochs);
#endif
    
    if (show) REPORT("Network Training Status:");
    
    // Performing the training.
    unsigned numFails = 0;
    REAL trnError, valError;
    unsigned dispCounter = 0;
    
    for (unsigned epoch=0; epoch<nEpochs; epoch++)
    {
      trnError = train->trainNetwork();
      valError = train->valNetwork();

      // Saving the best weight result.
      if (train->isBestNetwork(valError))
      {
        net->saveBestTrain();
        //Reseting the numFails counter.
        numFails = 0;
      }
      else numFails++;

      //Showing partial results at every "show" epochs (if show != 0).
      if (show)
      {
        if (!dispCounter) train->showTrainingStatus(epoch, trnError, valError);
        dispCounter = (dispCounter + 1) % show;
      }

      //Saving the training evolution info.
      train->saveTrainInfo(epoch, trnError, valError);

      //Updating the weight and bias matrices.
      net->updateWeights(batchSize);
      
      if (numFails == maxFail)
      {
        if (show) REPORT("Maximum number of failures reached. Finishing training...");
        break;
      }
    }

    // Generating a copy of the network structure passed as input.
    ret[OUT_NET_IDX] = mxDuplicateArray(netStr);
  
    //Saving the training results.
    net->flushBestTrainWeights(ret[OUT_NET_IDX]);
    
    //Returning the training evolution info.
    train->flushTrainInfo(ret[OUT_EPOCH_IDX], ret[OUT_TRN_ERROR_IDX], ret[OUT_VAL_ERROR_IDX]);
    
    //Deleting the allocated memory.
    delete net;
    delete train;
    if (show) REPORT("Training process finished!");
  }
  catch(bad_alloc xa)
  {
    FATAL("Error on allocating memory!");
    if (net) delete net;
    if (train) delete train;
  }
  catch (const char *msg)
  {
    FATAL(msg);
    if (net) delete net;
    if (train) delete train;
  }
}
