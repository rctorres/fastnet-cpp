#include "fastnet/training/PatternRec.h"

PatternRecognition::PatternRecognition(FastNet::Backpropagation *net, const mxArray *inTrn, 
                                        const mxArray *inVal, const mxArray *inTst,  
                                        const bool usingSP, const unsigned bSize) 
                                        : Training(net, bSize)
{
  DEBUG1("Starting a Pattern Recognition Training Object");
  
  hasTstData = inTst != NULL;
  useSP = usingSP;
  if (useSP)
  {
    bestGoalSP = 0.;
    DEBUG2("I'll use SP validating criterium.");
  }
  else DEBUG2("I'll NOT use SP validating criterium.");
  
  numPatterns = mxGetN(inTrn);
  DEBUG2("Number of patterns: " << numPatterns);
  outputSize = (numPatterns == 2) ? 1 : numPatterns;
  
  //The last 2 parameters for the training case will not be used by the function, so, there is no 
  //problem passing the corresponding validation variables to this first function call.
  DEBUG2("Allocating memory for the training data.");
  allocateDataset(inTrn, true, inTrnList, epochValOutputs, numValEvents);
  DEBUG2("Allocating memory for the validation data.");
  allocateDataset(inVal, false, inValList, epochValOutputs, numValEvents);
  if (hasTstData)
  {
    DEBUG2("Allocating memory for the testing data.");
    allocateDataset(inTst, false, inTstList, epochTstOutputs, numTstEvents);
  }
  //Creating the targets for each class (maximum sparsed oututs).
  targList = new const REAL* [numPatterns];  
  for (unsigned i=0; i<numPatterns; i++)
  {
    REAL *target = new REAL [outputSize];
    for (unsigned j=0; j<outputSize; j++) target[j] = -1;
    target[i] = 1;
    //Saving the target in the list.
    targList[i] = target;    
  }
  
  DEBUG2("Input events dimension: " << inputSize);
  DEBUG2("Output events dimension: " << outputSize);
};


void PatternRecognition::allocateDataset(const mxArray *dataSet, const bool forTrain, 
                                         const REAL **&inList, REAL **&out, unsigned *&nEv)
{
  inList = new const REAL* [numPatterns];

  if (!forTrain)
  {
    nEv = new unsigned [numPatterns];
    if (useSP) out = new REAL* [numPatterns];
  }
  
  for (unsigned i=0; i<numPatterns; i++)
  {
    const mxArray *patData = mxGetCell(dataSet, i);
    inputSize = mxGetM(patData);
    inList[i] = static_cast<REAL*>(mxGetData(patData));

    if (forTrain) dmTrn.push_back(new DataManager(mxGetN(patData)));
    else
    {
      nEv[i] = static_cast<unsigned>(mxGetN(patData));
      if (useSP) out[i] = new REAL [nEv[i]];
    }
  }
}

void PatternRecognition::deallocateDataset(const bool forTrain, const REAL **&inList, REAL **&out, unsigned *&nEv)
{
  for (unsigned i=0; i<numPatterns; i++)
  {
    if (forTrain) delete dmTrn[i];
    else if (useSP) delete [] out[i];
  }

  delete [] inList;
  if (!forTrain)
  {
    delete [] nEv;
    if (useSP) delete [] out;
  }
}


PatternRecognition::~PatternRecognition()
{
  //The last 2 parameters for the training case will not be used by the function, so, there is no 
  //problem passing the corresponding validation variables to this first function call.
  deallocateDataset(true, inTrnList, epochValOutputs, numValEvents);
  deallocateDataset(false, inValList, epochValOutputs, numValEvents);
  if (hasTstData) deallocateDataset(false, inTstList, epochTstOutputs, numTstEvents);
  for (unsigned i=0; i<numPatterns; i++) delete [] targList[i];
  delete [] targList;
};


REAL PatternRecognition::sp(const unsigned *nEvents, REAL **epochOutputs)
{
  unsigned TARG_SIGNAL, TARG_NOISE;
  
  //We consider that our signal has target output +1 and the noise, -1. So, the if below help us
  //figure out which class is the signal.
  if (targList[0][0] > targList[1][0]) // target[0] is our signal.
  {
    TARG_NOISE = 1;
    TARG_SIGNAL = 0;    
  }
  else //target[1] is the signal.
  {
    TARG_NOISE = 0;
    TARG_SIGNAL = 1;
  }

  const REAL *signal = epochOutputs[TARG_SIGNAL];
  const REAL *noise = epochOutputs[TARG_NOISE];
  const REAL signalTarget = targList[TARG_SIGNAL][0];
  const REAL noiseTarget = targList[TARG_NOISE][0];
  const int numSignalEvents = static_cast<int>(nEvents[TARG_SIGNAL]);
  const int numNoiseEvents = static_cast<int>(nEvents[TARG_NOISE]);
  const REAL RESOLUTION = 0.01;
  REAL maxSP = -1.;
  int i;
  int chunk = chunkSize;


  for (REAL pos = noiseTarget; pos < signalTarget; pos += RESOLUTION)
  {
    REAL sigEffic = 0.;
    REAL noiseEffic = 0.;
    unsigned se, ne;
    
    #pragma omp parallel shared(signal, noise, sigEffic, noiseEffic) private(i,se,ne)
    {
      se = ne = 0;
      
      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numSignalEvents; i++) if (signal[i] >= pos) se++;
      
      #pragma omp critical
      sigEffic += static_cast<REAL>(se);

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numNoiseEvents; i++) if (noise[i] < pos) ne++;
      
      #pragma omp critical
      noiseEffic += static_cast<REAL>(ne);
    }
    
    sigEffic /= static_cast<REAL>(numSignalEvents);
    noiseEffic /= static_cast<REAL>(numNoiseEvents);
    
    //Using normalized SP calculation.
    const REAL sp = ((sigEffic + noiseEffic) / 2) * sqrt(sigEffic * noiseEffic);
    if (sp > maxSP) maxSP = sp;
  }
  
  return maxSP;
};


void PatternRecognition::getNetworkErrors(const REAL **inList, const unsigned *nEvents,
                                           REAL **epochOutputs, REAL &mseRet, REAL &spRet)
{
  DEBUG2("Starting validation process for an epoch.");
  REAL gbError = 0.;
  FastNet::Backpropagation **nv = netVec;
  int totEvents = 0;
  
  for (unsigned pat=0; pat<numPatterns; pat++)
  {
    totEvents += nEvents[pat];
 
    const REAL *target = targList[pat];
    const REAL *input = inList[pat];
    const REAL *output;
    const int numEvents = nEvents[pat];
    REAL error = 0.;
    int i, thId;
    int chunk = chunkSize;

    REAL *outList = (useSP) ? epochOutputs[pat] : NULL;
    
    DEBUG3("Applying performance calculation for pattern " << pat << ".");
    
    #pragma omp parallel shared(input,target,chunk,nv,gbError,pat) private(i,thId,output,error)
    {
      thId = omp_get_thread_num();
      error = 0.;

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numEvents; i++)
      {
        error += nv[thId]->applySupervisedInput(&input[i*inputSize], target, output);
        if (useSP) outList[i] = output[0];
      }

      #pragma omp critical
      gbError += error;
    }
  }

  mseRet = gbError / static_cast<REAL>(totEvents);
  if (useSP)  spRet = sp(nEvents, epochOutputs);
};


REAL PatternRecognition::trainNetwork()
{
  DEBUG2("Starting training process for an epoch.");
  REAL gbError = 0;
  FastNet::Backpropagation **nv = netVec;
  updateNetworks();

  for(unsigned pat=0; pat<numPatterns; pat++)
  {
    //wFactor will allow each pattern to have the same relevance, despite the number of events it contains.
    const REAL *target = targList[pat];
    const REAL *input = inTrnList[pat];
    const REAL *output;
    REAL error = 0.;
    int i, thId;
    int chunk = chunkSize;
    unsigned pos = 0;
    DataManager *dm = dmTrn[pat];

    DEBUG3("Applying training set for pattern " << pat << ".");

    #pragma omp parallel shared(input,target,chunk,nv,gbError,pat,dm) private(i,thId,output,error,pos)
    {
      thId = omp_get_thread_num();
      error = 0.;

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<batchSize; i++)
      {
        #pragma omp critical
        pos = dm->get();

        error += nv[thId]->applySupervisedInput(&input[pos*inputSize], target, output);
        //Calculating the weight and bias update values.
        nv[thId]->calculateNewWeights(output, target);
      }

      #pragma omp critical
      gbError += error;
    }
  }

  updateGradients();
  return (gbError / static_cast<REAL>(numPatterns*batchSize));  
};
  
void PatternRecognition::checkSizeMismatch() const
{
  if (inputSize != (*net)[0]) throw "Input training or validating data do not match the network input layer size!";
};


void PatternRecognition::showInfo(const unsigned nEpochs) const
{
  REPORT("TRAINING DATA INFORMATION (Pattern Recognition Optimized Network)");
  REPORT("Number of Epochs          : " << nEpochs);
  REPORT("Using SP Stopping Criteria      : " << (useSP) ? "true" : "false");
};

void PatternRecognition::isBestNetwork(const REAL currMSEError, const REAL currSPError, bool &isBestMSE, bool &isBestSP)
{
  //Knowing whether we have a better network, according to the MSE validation criterium.
  Training::isBestNetwork(currMSEError, currSPError, isBestMSE, isBestSP);

  //Knowing whether we have a better network, according to the SP validation criterium.  
  if (useSP)
  {
    if (currSPError > bestGoalSP)
    {
      bestGoalSP = currSPError;
      isBestSP = true;
    }
    else isBestSP = false;
  }
};

void PatternRecognition::showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError)
{
  if (useSP) {REPORT("Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << " SP (val) = " << valError)}
  else Training::showTrainingStatus(epoch, trnError, valError);
};
