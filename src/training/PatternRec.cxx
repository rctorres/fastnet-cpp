#include "fastnet/training/PatternRec.h"

PatternRecognition::PatternRecognition(FastNet::Backpropagation *net, const mxArray *inTrn, const mxArray *inVal, const bool usingSP, const unsigned bSize) : Training(net, bSize)
{
  DEBUG1("Starting a Pattern Recognition Training Object");
  if (mxGetN(inTrn) != mxGetN(inVal)) throw "Number of training and validating patterns are not equal";
  
  useSP = usingSP;
  if (useSP)
  {
    bestGoal = 0.;
    DEBUG2("I'll use SP validating criterium.");
  }
  else DEBUG2("I'll NOT use SP validating criterium.");
  
  numPatterns = mxGetN(inTrn);
  DEBUG2("Number of patterns: " << numPatterns);
  outputSize = (numPatterns == 2) ? 1 : numPatterns;
  inTrnList = new const REAL* [numPatterns];
  inValList = new const REAL* [numPatterns];
  targList = new const REAL* [numPatterns];
  if (useSP) epochValOutputs = new REAL* [numPatterns];
  
  for (unsigned i=0; i<numPatterns; i++)
  {
    const mxArray *patTrnData = mxGetCell(inTrn, i);
    const mxArray *patValData = mxGetCell(inVal, i);    

    //Checking whether the dimensions are ok.
    if ( mxGetM(patTrnData) != mxGetM(patValData) ) throw "Input training and validating events dimension does not match!";
    if ( (i) and (mxGetM(patTrnData) != inputSize)) throw "Events dimension between patterns does not match!";
    else inputSize = mxGetM(patTrnData);

    //Getting the desired values.    
    inTrnList[i] = static_cast<REAL*>(mxGetData(patTrnData));
    inValList[i] = static_cast<REAL*>(mxGetData(patValData));
    dmTrn.push_back(new DataManager(mxGetN(patTrnData)));
    dmVal.push_back(new DataManager(mxGetN(patValData)));
    if (useSP) epochValOutputs[i] = new REAL [outputSize*batchSize];
    DEBUG2("Number of training events for pattern " << i << ": " << mxGetN(patTrnData));
    DEBUG2("Number of validating events for pattern " << i << ": " << mxGetN(patValData));
    
    //Generating the desired output for each pattern for maximum sparsed outputs.
    REAL *target = new REAL [outputSize];
    for (unsigned j=0; j<outputSize; j++) target[j] = -1;
    target[i] = 1;
    //Saving the target in the list.
    targList[i] = target;    
  }
  
  DEBUG2("Input events dimension: " << inputSize);
  DEBUG2("Output events dimension: " << outputSize);
};

PatternRecognition::~PatternRecognition()
{
  for (unsigned i=0; i<numPatterns; i++)
  {
    delete dmTrn[i];
    delete dmVal[i];
    delete [] inTrnList[i];
    delete [] inValList[i];
    delete [] targList[i];
    if (useSP) delete [] epochValOutputs[i];
  }
  delete [] inTrnList;
  delete [] inValList;
  delete [] targList;
  if (useSP) delete [] epochValOutputs;
};

REAL PatternRecognition::sp()
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

  const REAL *signal = epochValOutputs[TARG_SIGNAL];
  const REAL *noise = epochValOutputs[TARG_NOISE];
  const REAL signalTarget = targList[TARG_SIGNAL][0];
  const REAL noiseTarget = targList[TARG_NOISE][0];
  const REAL RESOLUTION = 0.001;
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
      for (i=0; i<batchSize; i++) if (signal[i] >= pos) se++;
      
      #pragma omp critical
      sigEffic += static_cast<REAL>(se);

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<batchSize; i++) if (noise[i] < pos) ne++;
      
      #pragma omp critical
      noiseEffic += static_cast<REAL>(ne);
    }
    
    sigEffic /= static_cast<REAL>(batchSize);
    noiseEffic /= static_cast<REAL>(batchSize);
    
    //Using normalized SP calculation.
    const REAL sp = ((sigEffic + noiseEffic) / 2) * sqrt(sigEffic * noiseEffic);
    if (sp > maxSP) maxSP = sp;
  }
  
  return maxSP;
};

REAL PatternRecognition::valNetwork()
{
  DEBUG2("Starting validation process for an epoch.");
  REAL gbError = 0.;
  FastNet::Backpropagation **nv = netVec;
  
  for (unsigned pat=0; pat<numPatterns; pat++)
  {
    const REAL *target = targList[pat];
    const REAL *input = inValList[pat];
    const REAL *output;
    unsigned pos = 0;
    REAL error = 0.;
    int i, thId;
    int chunk = chunkSize;
    DataManager *dm = dmVal[pat];

    REAL *outList = (useSP) ? epochValOutputs[pat] : NULL;
    
    DEBUG3("Applying validation set for pattern " << pat << ". Weighting factor to use: " << wFactor);
    
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
        if (useSP) outList[i] = output[0];
      }
      
      #pragma omp critical
      gbError += error;
    }
  }

  return (useSP) ? sp() : (gbError / static_cast<REAL>(numPatterns*batchSize));
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

    DEBUG3("Applying training set for pattern " << pat << ". Weighting factor to use: " << wFactor);

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

bool PatternRecognition::isBestNetwork(const REAL currError)
{
  if (useSP)
  {
    if (currError > bestGoal)
    {
      bestGoal = currError;
      return true;
    }
    return false;
  }
  
  //Otherwise we use the standard MSE method.
  return Training::isBestNetwork(currError);
};

void PatternRecognition::showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError)
{
  if (useSP) {REPORT("Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << " SP (val) = " << valError)}
  else Training::showTrainingStatus(epoch, trnError, valError);
};
