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

#include "fastnet/reporter/Reporter.h"
#include "fastnet/neuralnet/backpropagation.h"
#include "fastnet/neuralnet/rprop.h"
#include "fastnet/events/matevents.h"
#include "fastnet/defines.h"
#include "fastnet/events/mxhandler.h"

using namespace std;
using namespace FastNet;

/// Specifies the correct number of arguments if the neural network will just be trained.
/**
 If the user specifies only 5 input parameters, the network will be trained and tested
 with an epoch size (training and testing) corresponding to the total amount of 
 training and testing data.
*/
const unsigned NUM_ARGS_FULL_EPOCH = 5;

/// Specifies the correct number of arguments if the neural network will be trained and tested with specified epochs size.
/**
 If the user specifies the 6 input parameters, the network will be trained and tested
 considering, in each epoch, the number of events specified in the last 2 parameters.
 */
const unsigned NUM_ARGS_PARTIAL_EPOCH = 6;

/// Index, in the arguments list, of the neural network structure.
const unsigned NET_STR_IDX = 0;

/// Index, in the arguments list, of the input training events.
const unsigned IN_TRN_IDX = 1;

/// Index, in the arguments list, of the output training events.
const unsigned OUT_TRN_IDX = 2;

/// Index, in the arguments list, of the input testing events.
const unsigned IN_TST_IDX = 3;

/// Index, in the arguments list, of the output testing events.
const unsigned OUT_TST_IDX = 4;

/// Index, in the arguments list, of the training epoch size.
const unsigned TRN_EPOCH_SIZE_IDX = 5;

/// Index, in the return vector, of the network structure after training.
const unsigned OUT_NET_IDX = 0;

/// Index, in the return vector, of the vector containing the epoch values.
const unsigned OUT_EPOCH_IDX = 1;

/// Index, in the return vector, of the vector containing the training error obtained in each epoch.
const unsigned OUT_TRN_ERROR_IDX = 2;

//Index, in the return vector, of the vector containing the testing error obtained in each epoch.
const unsigned OUT_TST_ERROR_IDX = 3;

//This struct will hold the traiing info to be ruterned to the user.
struct TrainData
{
  REAL epoch;;
  REAL trnError;
  REAL tstError;
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
  @param[out] trnList The list where to save the info.
 */
 void saveTrainInfo(unsigned epoch, REAL trnError, REAL tstError, list<TrainData> &trnList)
 {
   TrainData trainData;
   trainData.epoch = (REAL) epoch;
   trainData.trnError = trnError;
   trainData.tstError = tstError;
   trnList.push_back(trainData);
 }


/// Flush trining evolution info to Matlab vectors.
/**
 Since this class, in order to optimize speed, saves the
 training information (epochs and errors) values into memory, at the end, if the user wants
 to save the final values, this method must be called. It will
 save these values stored in the linked list in Matlab vectors.
 @param[in] trnList The list where the training evolution info is stored.
 @param[out] epoch A vector containing the epochs values.
 @param[out] trnError A vector containing the training error obtained in each epoch.
 @param[out] tstErrorA vector containing the testing error obtained in each epoch.
*/
void flushTrainInfo(const list<TrainData> &trnList, mxArray *&epoch, mxArray *&trnError, mxArray *&tstError)
{
  const unsigned size = trnList.size();  
  epoch = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);
  trnError = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);
  tstError = mxCreateNumericMatrix(1, size, REAL_TYPE, mxREAL);

  REAL *ep = static_cast<REAL*>(mxGetData(epoch));
  REAL *trn = static_cast<REAL*>(mxGetData(trnError));
  REAL *tst = static_cast<REAL*>(mxGetData(tstError));
  
  for (list<TrainData>::const_iterator itr = trnList.begin(); itr != trnList.end(); itr++)
  {
    *ep++ = itr->epoch;
    *trn++ = itr->trnError;
    *tst++ = itr->tstError;
  }
}


/// Calculates the SP product.
/**
 Calculates the SP product. This method will run through the dynamic range of the outputs,
 calculating the SP product in each lambda value. Returning, at the end, the maximum SP
 product obtained.
 @param[in] target the list of output (target) for each pattern.
 @param[in] output Will contain the outputs for each pattern. The method does NOT allocate any 
 memory, this vector must come ready to store all outut values, for all patterns.
 @return The maximum SP value obtained.
*/
REAL sp(const vector<REAL*> &target, const vector< vector<REAL*>* > &output)
{
  const REAL RESOLUTION = 0.001;
  unsigned TARG_SIGNAL, TARG_NOISE;
  
  //We consider that our signal has target output +1 and the noise, -1. So, the if below help us
  //figure out which class is the signal.
  if (target[0][0] > target[1][0]) // target[0] is our signal.
  {
    TARG_NOISE = 1;
    TARG_SIGNAL = 0;
  }
  else //target[1] is the signal.
  {
    TARG_NOISE = 0;
    TARG_SIGNAL = 1;
  }
  
  REAL maxSP = -1.;
  for (REAL pos = target[TARG_NOISE][0]; pos < target[TARG_SIGNAL][0]; pos += RESOLUTION)
  {
    REAL sigEffic = 0.;
    for (vector<REAL*>::const_iterator itr = output[TARG_SIGNAL]->begin(); itr != output[TARG_SIGNAL]->end(); itr++)
    {
      if ((*itr)[0] >= pos) sigEffic++;
    }
    sigEffic /= static_cast<REAL>(output[TARG_SIGNAL]->size());

    REAL noiseEffic = 0.;
    for (vector<REAL*>::const_iterator itr = output[TARG_NOISE]->begin(); itr != output[TARG_NOISE]->end(); itr++)
    {
      if ((*itr)[0] < pos) noiseEffic++;
    }
    noiseEffic /= static_cast<REAL>(output[TARG_NOISE]->size());

    //Using normalized SP calculation.
    const REAL sp = ((sigEffic + noiseEffic) / 2) * sqrt(sigEffic * noiseEffic);
    if (sp > maxSP) maxSP = sp;
  }
  
  return maxSP;
}



/// Applies the testing set for the network's testing.
/**
 This method takes the one or more testing events (input and targets) and presents them
 to the network. At the end, the mean training error is returned. Since it is a testing function,
 the network is not modified, and no updating weights values are calculated. This method only
 presents the testing sets and calculates the mean testing error obtained.
 @param[in] net the network class that the events will be presented to. The internal parameters
 of this class are not modified inside this method, since it is only a network testing process.
 @param[in] inData the input testing events.
 @param[in] outData the output (target) testing events.
 @return The mean testing error obtained after the entire training set is presented to the network.
*/
REAL testNetwork(NeuralNetwork *net, MatEvents *inData, MatEvents *outData)
{
  REAL gbError = 0.;
  const REAL *out;
  
  for (unsigned i=0; i<inData->getNumEvents(); i++)
  {
    // Getting the next input and target pair.
    const REAL *input = inData->readEvent(i);
    const REAL *target = outData->readEvent(i);
    gbError += net->applySupervisedInput(input, target, out);
  }

  return (gbError / static_cast<REAL>(inData->getNumEvents()));
}


/// Applies the training set for the network's training.
/**
 This method takes the one or more training events (input and targets) and presents them
 to the network, calculating the new mean (if batch training is being used) update values 
 after each input-output pair is presented. At the end, the mean training error is returned.
 @param[in] net the network class that the events will be presented to. At the end,
 this class is modificated, as it will contain the mean values of \f$\Delta w\f$ and \f$\Delta b\f$ obtained
 after the entire training set has been presented, but the weights are not updated at the 
 end of this function. To actually update the weights, the user must call the proper
 class's method for that.
 @param[in] inData the input training events.
 @param[in] outData the output (target) training events.
 @param[in] epochSize The number of events to apply to the network.
 @return The mean training error obtained after the entire training set is presented to the network.
*/
REAL trainNetwork(NeuralNetwork *net, MatEvents *inData, MatEvents *outData, const unsigned epochSize)
{
  unsigned evIndex;
  REAL gbError = 0.;
  const REAL *output;

  for (unsigned i=0; i<epochSize; i++)
  {
    // Getting the next input and target pair.
    const REAL *input = inData->readRandomEvent(evIndex);
    const REAL *target = outData->readEvent(evIndex);
    gbError += net->applySupervisedInput(input, target, output);

    //Calculating the weight and bias update values
    net->calculateNewWeights(output, target);
  }
  return (gbError / static_cast<REAL>(epochSize));
}


/// Applies the testing set of each pattern for the network's testing.
/**
 This method takes the one or more pattern's testing events (input and targets) and presents them
 to the network. At the end, the mean training error is returned. Since it is a testing function,
 the network is not modified, and no updating weights values are calculated. This method only
 presents the testing sets and calculates the mean testing error obtained.
 @param[in] net the network class that the events will be presented to. The internal parameters
 of this class are not modified inside this method, since it is only a network testing process.
 @param[in] inList the list of input testing events for each pattern.
 @param[in] target the list of output (target) for each pattern.
 @param[out] output Will contain the outputs for each pattern. The method does NOT allocate any 
 memory, this vector must come ready to store all outut values, for all patterns.
 @return The mean testing error obtained after the entire training set is presented to the network.
*/
REAL testNetwork(NeuralNetwork *net, const vector<MatEvents*> &inList, const vector<REAL*> &target, vector< vector<REAL*>* > &output)
{
  const REAL *out;
  REAL gbError = 0;
  unsigned totEvents = 0;
  const unsigned outSize = (*net)[net->getNumLayers()-1];
  
  for (unsigned pat=0; pat<inList.size(); pat++)
  {
    for (unsigned i=0; i<inList[pat]->getNumEvents(); i++)
    {
      // Getting the next input and target pair.
      const REAL *input = inList[pat]->readEvent(i);
      gbError += net->applySupervisedInput(input, target[pat], out);
      memcpy(output[pat]->at(i), out, outSize*sizeof(REAL));
    }
    
    totEvents += inList[pat]->getNumEvents();
  }

  return (gbError / static_cast<REAL>(totEvents));
}


/// Applies the training set of each pattern for the network's training.
/**
 This method takes the one or more patterns training events (input and targets) and presents them
 to the network, calculating the new mean (if batch training is being used) update values 
 after each input-output pair of each individual pattern is presented. At the end, the mean training error is returned.
 @param[in] net the network class that the events will be presented to. At the end,
 this class is modificated, as it will contain the mean values of \f$\Delta w\f$ and \f$\Delta b\f$ obtained
 after the entire training set has been presented, but the weights are not updated at the 
 end of this function. To actually update the weights, the user must call the proper
 class's method for that.
 @param[in] inList the input training events for each pattern.
 @param[in] target the output (target) training events for each pattern.
 @param[in] epochSize The number of events in each pattern to apply to the network.
 @return The mean training error obtained after the entire training of each pattern set is presented to the network.
*/
REAL trainNetwork(NeuralNetwork *net, const vector<MatEvents*> &inList, const vector<REAL*> &target, const vector<unsigned> &epochSize)
{
  unsigned evIndex;
  const REAL *output;
  REAL gbError = 0;
  unsigned totEvents = 0;
  
  for(unsigned pat=0; pat<inList.size(); pat++)
  {
    for (unsigned i=0; i<epochSize[pat]; i++)
    {
      // Getting the next input and target pair.
      const REAL *input = inList[pat]->readRandomEvent(evIndex);
      gbError += net->applySupervisedInput(input, target[pat], output);
      //Calculating the weight and bias update values.
      net->calculateNewWeights(output, target[pat], epochSize[pat], inList.size());    
    }
    
    totEvents += epochSize[pat];
  }

  return (gbError / static_cast<REAL>(totEvents));  
}

REAL greaterThan(REAL a, REAL b) {return (a>b);}
REAL smallerThan(REAL a, REAL b) {return (a<b);}

/// Matlab 's main function.
void mexFunction(int nargout, mxArray *ret[], int nargin, const mxArray *args[])
{
  MatEvents *inTrnData, *outTrnData, *inTstData, *outTstData;
  vector<MatEvents*> inTrnList, inTstList;
  vector<REAL*> outList;
  vector< vector<REAL*>* > epochTstOutputs;
  NeuralNetwork *net = NULL;

  try
  {  
    //Verifying if the number of input parameters is ok.
    if ( (nargin != NUM_ARGS_FULL_EPOCH) && (nargin != NUM_ARGS_PARTIAL_EPOCH) )
    {
      throw "Incorrect number of arguments! See help for information!";
    }

    //Reading the configuration structure
    const mxArray *netStr = args[NET_STR_IDX];
    
    //Opening the events handlers.
    bool patRecNet;
    unsigned numPat;
    inTrnData = outTrnData = inTstData = outTstData = NULL;
    if ( (!mxIsEmpty(args[OUT_TRN_IDX])) && (!mxIsEmpty(args[OUT_TST_IDX])) )
    {
      patRecNet = false;
      inTrnData = new MatEvents (args[IN_TRN_IDX]);
      outTrnData = new MatEvents (args[OUT_TRN_IDX]);
      inTstData = new MatEvents (args[IN_TST_IDX]);
      outTstData = new MatEvents (args[OUT_TST_IDX]);
    }
    else // It is a pattern recognition network.
    {
      if (mxGetN(args[IN_TRN_IDX]) != mxGetN(args[IN_TST_IDX])) throw "Number of training and testing patterns are not equal";
      patRecNet = true;
      numPat = mxGetN(args[IN_TRN_IDX]);
      for (unsigned i=0; i<numPat; i++)
      {
        //Getting the training data for each pattern.
        inTrnList.push_back(new MatEvents (mxGetCell(args[IN_TRN_IDX], i)));
        //Getting the test data for each pattern.
        inTstList.push_back(new MatEvents (mxGetCell(args[IN_TST_IDX], i)));
        //Generating the desired output for each pattern for maximum sparsed outputs.
        REAL *output = new REAL [numPat];
        for (unsigned j=0; j<numPat; j++) output[j] = -1;
        output[i] = 1;
        //Saving the target in the list.
        outList.push_back(output);
        
        //Allocating space for the generated outputs...
        vector<REAL*> *aux = new vector<REAL*>;
        for (unsigned j=0; j<inTstList[i]->getNumEvents(); j++) aux->push_back(new REAL [numPat]);
        epochTstOutputs.push_back(aux);
      }
    }
    
    // Determining if we will use all the events in each epoch or not.
    vector<unsigned> trnEpochList;
    const unsigned trnEpochSize = (nargin == NUM_ARGS_FULL_EPOCH) ? mxGetN(args[IN_TRN_IDX]) : static_cast<unsigned>(mxGetScalar(args[TRN_EPOCH_SIZE_IDX]));
    if (patRecNet)
    {
      if (nargin == NUM_ARGS_FULL_EPOCH)
      {
        for (unsigned i=0; i<numPat; i++) trnEpochList.push_back(mxGetN(mxGetCell(args[IN_TRN_IDX], i)));
        // If the number of patterns per epochs was not specified, the number of epochs for each
        // class will be the size of the pattern with the largest amount of events.
        const unsigned maxEpochSize = *max_element(trnEpochList.begin(), trnEpochList.end());
        for (unsigned i=0; i<numPat; i++) trnEpochList[i] = maxEpochSize;
      }
      else
      {
        if ( (mxGetN(args[TRN_EPOCH_SIZE_IDX]) != mxGetN(args[IN_TRN_IDX])) ) 
             throw "Number of patterns in the training events per epoch vector are not the same as the total number of patterns!";

        const double *numTrnEvEp = mxGetPr(args[TRN_EPOCH_SIZE_IDX]);
        for (unsigned i=0; i<numPat; i++) trnEpochList.push_back(static_cast<unsigned>(numTrnEvEp[i]));
      }
    }

    //Selecting the training type by reading the training agorithm.    
    const string trnType = mxArrayToString(mxGetField(netStr, 0, "trainFcn"));
    if (trnType == TRAINRP_ID)
    {
      net = new RProp(netStr);
      REPORT("Starting Resilient Backpropagation training...");
    }
    else if (trnType == TRAINGD_ID)
    {
      net = new Backpropagation(netStr);
      REPORT("Starting Gradient Descendent training...");
    }
    else throw "Invalid training algorithm option!";
    
    #ifdef DEBUG
    {
      ostringstream auxStr;
      net->showInfo(auxStr);
      DEBUG0(auxStr.str());
    }
    #endif
    
    //Reading the showing period, epochs and max_fail.
    const mxArray *trnParam =  mxGetField(netStr, 0, "trainParam");
    const unsigned nEpochs = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "epochs")));
    const unsigned show = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "show")));
    const unsigned maxFail = static_cast<unsigned>(mxGetScalar(mxGetField(trnParam, 0, "max_fail")));

    //Checking if the training and testing input data sizes match the network's input layer.
    if (!patRecNet)
    {
      if ( (mxGetM(args[IN_TRN_IDX]) != (*net)[0]) || (mxGetM(args[IN_TST_IDX]) != (*net)[0]) )
        throw "Input training or testing data do not match the network input layer size!";
    }
    else
    {
      for (unsigned i=0; i<numPat; i++)
      {
        if ( (mxGetM(mxGetCell(args[IN_TRN_IDX],i)) != (*net)[0]) || (mxGetM(mxGetCell(args[IN_TST_IDX],i)) != (*net)[0]) )
          throw "Input training or testing data do not match the network input layer size!";
      }
    }
    
    //Checking if the training and testing output data sizes match the network's output layer.
    if (!patRecNet)
    {
      if ( (mxGetM(args[OUT_TRN_IDX]) != (*net)[net->getNumLayers()-1]) || (mxGetM(args[OUT_TST_IDX]) != (*net)[net->getNumLayers()-1]) )
        throw "Output training or testing data do not match the network output layer size!";
    }
    else
    {
      if ( (numPat != (*net)[net->getNumLayers()-1]) && ( (numPat != 2) || ((*net)[net->getNumLayers()-1] != 1) ) )
        throw "Number of patterns does not match the number of nodes in the output layer!";
    }

    //Getting whether we will use SP stoping criteria.
    const mxArray *usingSP = mxGetField(mxGetField(netStr, 0, "userdata"), 0, "useSP");
    const bool useSP = static_cast<bool>(mxGetScalar(usingSP));

    //Displaying the training info before starting.
    if (!patRecNet)
    {
      REPORT("TRAINING DATA INFORMATION (Regular Network)");
      REPORT("Number of Epochs                    : " << nEpochs);
      REPORT("Number of training events per epoch : " << trnEpochSize);
      REPORT("Total number of training events     : " << inTrnData->getNumEvents());
      REPORT("Total number of testing events      : " << inTstData->getNumEvents());
    }
    else
    {
      REPORT("TRAINING DATA INFORMATION (Pattern Recognition Specific Network)");
      REPORT("Number of Epochs                    : " << nEpochs);
      REPORT("Using SP Stopping Criteria          : " << useSP);
      for (unsigned i=0; i<numPat; i++)
      {
        REPORT("Information for pattern " << (i+1) << ":");
        REPORT("Number of training events per epoch : " << trnEpochList[i]);
        REPORT("Total number of training events     : " << inTrnList[i]->getNumEvents());
        REPORT("Total number of testing events      : " << inTstList[i]->getNumEvents());
      }
    }
    
    REPORT("Network Training Status:");
    
    // Performing the training.
    unsigned numFails = 0;
    REAL trnError, tstError;
    REAL minTstError;
    REAL maxSP = 0.;
    unsigned dispCounter = 0;
    string tstText;
    list<TrainData> trnEvolutionData;
    REAL (*comp)(REAL, REAL);
    
    if (patRecNet && useSP) // If using the SP criterium, we must aim at maximizing it.
    {
      minTstError = 0.;
      tstText = ", SP (test) = ";
      comp = greaterThan;
    }
    else // If it is MSE, we have to minimize it.
    {
      minTstError = static_cast<REAL>(5E20);
      tstText = ", mse (test) = ";
      comp = smallerThan;
    }
    
    for (unsigned epoch=0; epoch<nEpochs; epoch++)
    {
      if (!patRecNet)
      {
        trnError = trainNetwork(net, inTrnData, outTrnData, trnEpochSize);
        tstError = testNetwork(net, inTstData, outTstData);
      }
      else
      {
        trnError = trainNetwork(net, inTrnList, outList, trnEpochList);
        tstError = testNetwork(net, inTstList, outList, epochTstOutputs);
        if (useSP) tstError = sp(outList, epochTstOutputs);
      }

      // Saving the best weight result.
      if ((*comp)(tstError,minTstError))
      {
        net->saveBestTrain();
        minTstError = tstError;
        //Reseting the numFails counter.
        numFails = 0;
      }
      else numFails++;

      //Showing partial results at every "show" epochs (if show != 0).
      if (show)
      {
        if (!dispCounter)
        {
          REPORT("Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << tstText << tstError);
        }
        dispCounter = (dispCounter + 1) % show;
      }

      //Saving the training evolution info.
      saveTrainInfo(epoch, trnError, tstError, trnEvolutionData);

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
    flushTrainInfo(trnEvolutionData, ret[OUT_EPOCH_IDX], ret[OUT_TRN_ERROR_IDX], ret[OUT_TST_ERROR_IDX]);
    
    //Deleting the allocated memory.
    delete net;
    if (!patRecNet)
    {
      delete inTrnData;
      delete outTrnData;
      delete inTstData;
      delete outTstData;
    }
    else
    {
      for (unsigned i=0; i<numPat; i++)
      {
        delete inTrnList[i];
        delete inTstList[i];
        delete [] outList[i];
        for (unsigned j=0; j<epochTstOutputs[i]->size(); j++) delete [] epochTstOutputs[i]->at(j);
        delete epochTstOutputs[i];
      }
    }
    REPORT("Training process finished!");
  }
  catch(bad_alloc xa)
  {
    FATAL("Error on allocating memory!");
    if (net) delete net;
    if (inTrnData) delete inTrnData;
    if (outTrnData) delete outTrnData;
    if (inTstData) delete inTstData;
    if (outTstData) delete outTstData;
    for (unsigned i=0; i<inTrnList.size(); i++)  delete inTrnList[i];
    for (unsigned i=0; i<inTstList.size(); i++)  delete inTstList[i];
    for (unsigned i=0; i<outList.size(); i++)  delete outList[i];
  }
  catch (const char *msg)
  {
    FATAL(msg);
    if (net) delete net;
    if (inTrnData) delete inTrnData;
    if (outTrnData) delete outTrnData;
    if (inTstData) delete inTstData;
    if (outTstData) delete outTstData;
    for (unsigned i=0; i<inTrnList.size(); i++)  delete inTrnList[i];
    for (unsigned i=0; i<inTstList.size(); i++)  delete inTstList[i];
    for (unsigned i=0; i<outList.size(); i++)  delete outList[i];
  }
}
