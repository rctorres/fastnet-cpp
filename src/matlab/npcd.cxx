/** 
@file  npcd.cpp
@brief The Matlab's npcd function definition file.

This function implements the principal components of discrimination training algorithm.
The number of PCD extracted will be the number of nodes in the hidden layer. Example:
if you pass a network topology like 100x20x1. The algorithm will extract the first 20 PCDs.
*/

#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mex.h>

#include "fastnet/reporter/Reporter.h"
#include "fastnet/neuralnet/backpropagation.h"
#include "fastnet/neuralnet/rprop.h"
#include "fastnet/matlab/Standard.hxx"
#include "fastnet/matlab/StandardMT.hxx"
#include "fastnet/matlab/PatternRec.hxx"
#include "fastnet/matlab/PatternRecMT.hxx"

using namespace std;
using namespace FastNet;

//Maximum number of allowed training threads.
const unsigned MAX_NUM_THREADS = 16;

/// Index, in the arguments list, of the neural network structure.
const unsigned NET_STR_IDX = 0;

/// Index, in the arguments list, of the input training events.
const unsigned IN_TRN_IDX = 1;

/// Index, in the arguments list, of the output training events.
const unsigned OUT_TRN_IDX = 2;

/// Index, in the arguments list, of the input validating events.
/**
This value might have to be changed during execution, because the index of the
validating input events is 3 for the standart training case, but it is 2 for the
Pattern recognition network case.
*/
unsigned IN_VAL_IDX = 3;

/// Index, in the arguments list, of the output validating events.
const unsigned OUT_VAL_IDX = 4;

/// Number of threads to use.
/**
This value might have to be changed during execution, if using standart training or pattern 
recognition case.
*/
unsigned NUM_THREADS_IDX = 5;

/// Index, in the return vector, of the network structure after training.
const unsigned OUT_NET_IDX = 0;

/// Index, in the return vector, of the vector containing the epoch values.
const unsigned OUT_EPOCH_IDX = 1;

/// Index, in the return vector, of the vector containing the training error obtained in each epoch.
const unsigned OUT_TRN_ERROR_IDX = 2;

//Index, in the return vector, of the vector containing the validating error obtained in each epoch.
const unsigned OUT_VAL_ERROR_IDX = 3;


vector<unsigned> getNumEvents(const mxArray *dataStr)
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


/// Matlab 's main function.
void mexFunction(int nargout, mxArray *ret[], int nargin, const mxArray *args[])
{
  Backpropagation *net = NULL;
  Training *train = NULL;
  
  try
  {
    bool stdTrainingType, getNumThreads;
    //Verifying if the number of input parameters is ok.
    if ( (nargin == 6) || (nargin == 5) ) // Standart training (ST ou MT case)
    {
      IN_VAL_IDX = 3;
      NUM_THREADS_IDX = 5;
      stdTrainingType = true;
      getNumThreads = (nargin == 6);
    }
    else if ( (nargin == 4) || (nargin == 3) )
    {
      IN_VAL_IDX = 2;
      NUM_THREADS_IDX = 3;
      stdTrainingType = false;
      getNumThreads = (nargin == 4);
    }
    else throw "Incorrect number of arguments! See help for information!";

    //Reading the configuration structure
    const mxArray *netStr = args[NET_STR_IDX];

    //Selecting the training type by reading the training agorithm.    
    const string trnType = mxArrayToString(mxGetField(netStr, 0, "trainFcn"));
    if (trnType == TRAINRP_ID)
    {
      net = new RProp(netStr, getNumEvents(args[IN_TRN_IDX]));
      REPORT("Starting Resilient Backpropagation training...");
    }
    else if (trnType == TRAINGD_ID)
    {
      net = new Backpropagation(netStr, getNumEvents(args[IN_TRN_IDX]));
      REPORT("Starting Gradient Descendent training...");
    }
    else throw "Invalid training algorithm option!";

    //Getting the number of working threads.
    const unsigned numThreads = (getNumThreads) ? static_cast<unsigned>(mxGetScalar(args[NUM_THREADS_IDX])) : 1;
    if ( (!numThreads) || (numThreads > MAX_NUM_THREADS) ) throw "Invalid number of threads.";
    REPORT("Number of working threads is " << numThreads);

    //Creating the object for the desired training type.
    if (stdTrainingType)
    {
      if (numThreads == 1) train = new StandardTraining(args[IN_TRN_IDX], args[OUT_TRN_IDX], args[IN_VAL_IDX], args[OUT_VAL_IDX]);
      else train = new StandardTrainingMT(net, args[IN_TRN_IDX], args[OUT_TRN_IDX], args[IN_VAL_IDX], args[OUT_VAL_IDX], numThreads);
    }
    else // It is a pattern recognition network.
    {
      //Getting whether we will use SP stoping criteria.
      const mxArray *usingSP = mxGetField(mxGetField(netStr, 0, "userdata"), 0, "useSP");
      const bool useSP = static_cast<bool>(mxGetScalar(usingSP));
      if (numThreads == 1) train = new PatternRecognition(args[IN_TRN_IDX], args[IN_VAL_IDX], useSP);
      else train = new PatternRecognitionMT(net, args[IN_TRN_IDX], args[IN_VAL_IDX], useSP, numThreads);
    }
    
    //Reading the showing period, epochs and max_fail.
    const mxArray *trnParam =  mxGetField(netStr, 0, "trainParam");
    const unsigned nEpochs = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "epochs")));
    const unsigned show = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "show")));
    const unsigned maxFail = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "max_fail")));

    //Checking if the training and validating input data sizes match the network's input layer.
    train->checkSizeMismatch(net);

    //Displaying the training info before starting.
    train->showInfo(nEpochs);
    
    REPORT("Network Training Status:");

    const unsigned pcdLayer = 1; // The PCD projection layer is the first input layer.
    const unsigned numPCDs = (*net)[pcdLayer];

    for (unsigned pcd=0; pcd<numPCDs; pcd++)
    {
      REPORT("Extracting PCD #" << pcd);
      //Setting the network for PCD training by activating only the first node.
      net->setActiveNodes(pcdLayer, 0, pcd);
      DEBUG1("Active nodes goes from 0 to " << pcd);
      if (pcd) net->setFrozen(pcdLayer, pcd-1, true); // Freezing the last trained node.
      
#ifdef DEBUG
      for (unsigned i=0; i<numPCDs; i++)
        DEBUG1("Node " << i << " in layer " << pcdLayer << " is frozen? " << net->isFrozen(pcdLayer), i); 
#endif

      // Performing the training.
      unsigned numFails = 0;
      REAL trnError, valError;
      unsigned dispCounter = 0;
    
      for (unsigned epoch=0; epoch<nEpochs; epoch++)
      {
        trnError = train->trainNetwork(net);
        valError = train->valNetwork(net);

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
        net->updateWeights();
      
        if (numFails == maxFail)
        {
          REPORT("Maximum number of failures reached. Finishing training...");
          break;
        }
      }

      // Generating a copy of the network structure passed as input.
      ret[OUT_NET_IDX] = mxDuplicateArray(netStr);
  
      //Saving the training results.
      net->flushBestTrainWeights(ret[OUT_NET_IDX]);
    
      //Returning the training evolution info.
      train->flushTrainInfo(ret[OUT_EPOCH_IDX], ret[OUT_TRN_ERROR_IDX], ret[OUT_VAL_ERROR_IDX]);
    }
    
    //Deleting the allocated memory.
    delete net;
    delete train;
    REPORT("Training process finished!");
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
