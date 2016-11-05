#include "fastnet/training/PatternRec.h"

PatternRecognition::PatternRecognition(FastNet::Backpropagation *net, std::vector<DataManager*> *inTrn, 
                                        std::vector<DataManager*> *inVal,  
                                        const bool usingSP, const unsigned bSize,
                                        const REAL signalWeight, const REAL noiseWeight) 
                                        : Training(net, bSize)
{
  DEBUG1("Starting a Pattern Recognition Training Object");
  
  inTrnList = inTrn;
  inValList = inVal;
  
  // Initialize weights for SP calculation
  this->signalWeight = signalWeight;
  this->noiseWeight = noiseWeight;

  useSP = usingSP;
  if (useSP)
  {
    bestGoalSP = 0.;
    DEBUG2("I'll use SP validating criterium.");
  }
  else DEBUG2("I'll NOT use SP validating criterium.");
  
  //Allocating space for the network outputs if SP criteria is selected.
  if (useSP)
  {
    for (const auto &patData : (*inTrnList) )
    {
      epochValOutputs.push_back(new REAL[patData->numEvents()]);
    }
  }
  
  //Creating the targets for each class (maximum sparsed oututs).
  const auto numPatterns = inTrn->size();
  DEBUG2("Number of patterns: " << numPatterns);
  const auto outputSize = (numPatterns == 2) ? 1 : numPatterns;
  for (auto i=0; i<numPatterns; i++)
  {
    REAL *target = new REAL [outputSize];
    for (auto j=0; j<outputSize; j++) target[j] = -1;
    target[i] = 1;
    //Saving the target in the list.
    targList.push_back(target);    
  }
  
  DEBUG2("Input events dimension: " << (*inTrn)[0]->eventSize());
  DEBUG2("Output events dimension: " << outputSize);
};


PatternRecognition::~PatternRecognition()
{
  for (auto &v : epochValOutputs) delete [] v;
  for (auto &v : targList) delete [] v;
};


REAL PatternRecognition::sp(const std::vector<DataManager*> *inList, const std::vector<REAL*> &epochOutputs)
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
  const int numSignalEvents = static_cast<int>((*inList)[TARG_SIGNAL]->numEvents());
  const int numNoiseEvents = static_cast<int>((*inList)[TARG_NOISE]->numEvents());
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

    // Use weights for signal and noise efficiencies
    sigEffic *= signalWeight;
    noiseEffic *= noiseWeight;

    //Using normalized SP calculation.
    const REAL sp = ((sigEffic + noiseEffic) / 2) * sqrt(sigEffic * noiseEffic);
    if (sp > maxSP) maxSP = sp;
  }
  
  return sqrt(maxSP); // This sqrt is so that the SP value is in percent.
};


void PatternRecognition::getNetworkErrors(const std::vector<DataManager*> *inList,
                                           std::vector<REAL*> &epochOutputs, REAL &mseRet, REAL &spRet)
{
  REAL gbError = 0.;
  FastNet::Backpropagation **nv = netVec;
  int totEvents = 0;
  
  for (auto pat=0; pat<inList->size(); pat++)
  {
 
    const REAL *target = targList[pat];
    const DataManager *input = (*inList)[pat];
    const REAL *output;
    const int numEvents = input->numEvents();
    REAL error = 0.;
    int i, thId;
    int chunk = chunkSize;
    totEvents += numEvents;

    REAL *outList = (useSP) ? epochOutputs[pat] : NULL;
    
    DEBUG2("Applying performance calculation for pattern " << pat << " (" << numEvents << " events).");
    
    #pragma omp parallel shared(input,target,chunk,nv,gbError,pat) private(i,thId,output,error)
    {
      thId = omp_get_thread_num();
      error = 0.;

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numEvents; i++)
      {
        error += nv[thId]->applySupervisedInput((*input)[i], target, output);
        if (useSP) outList[i] = output[0];
      }

      #pragma omp critical
      gbError += error;
    }
  }

  mseRet = gbError / static_cast<REAL>(totEvents);
  if (useSP)  spRet = sp(inList, epochOutputs);
};


REAL PatternRecognition::trainNetwork()
{
  DEBUG2("Starting training process for an epoch.");
  REAL gbError = 0;
  FastNet::Backpropagation **nv = netVec;
  int totEvents = 0; // Holds the amount of events presented to the network.

  for(unsigned pat=0; pat<inTrnList->size(); pat++)
  {
    //wFactor will allow each pattern to have the same relevance, despite the number of events it contains.
    const REAL *target = targList[pat];
    DataManager *input = (*inTrnList)[pat];
    const REAL *output;
    REAL error = 0.;
    int i, thId;
    int chunk = chunkSize;
    unsigned pos = 0;

    const int nEvents = (batchSize) ? batchSize : input->numEvents();
    totEvents += nEvents;
    DEBUG2("Applying training set for pattern " << pat << " by randomly selecting " << nEvents << " events (out of " << input->numEvents() << ").");
    
    #pragma omp parallel shared(input,target,chunk,nv,gbError,pat) private(i,thId,output,error,pos)
    {
      thId = omp_get_thread_num();
      error = 0.;

      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<nEvents; i++)
      {
        #pragma omp critical
        pos = input->getNextEventIndex();

        error += nv[thId]->applySupervisedInput((*input)[pos], target, output);
        //Calculating the weight and bias update values.
        nv[thId]->calculateNewWeights(output, target);
      }

      #pragma omp critical
      gbError += error;
    }
  }

  updateGradients();
  updateWeights();
  return (gbError / static_cast<REAL>(totEvents));
};
  

void PatternRecognition::showInfo(const unsigned nEpochs) const
{
  REPORT("TRAINING DATA INFORMATION (Pattern Recognition Optimized Network)");
  REPORT("Number of Epochs          : " << nEpochs);
  REPORT("Using SP Stopping Criteria      : " << ((useSP) ? "true" : "false"));
};

void PatternRecognition::isBestNetwork(const REAL currMSEError, const REAL currSPError, ValResult &isBestMSE, ValResult &isBestSP)
{
  //Knowing whether we have a better network, according to the MSE validation criterium.
  Training::isBestNetwork(currMSEError, currSPError, isBestMSE, isBestSP);

  //Knowing whether we have a better network, according to the SP validation criterium.  
  if (useSP)
  {
    if (currSPError > bestGoalSP)
    {
      bestGoalSP = currSPError;
      isBestSP = BETTER;
    }
    else if (currSPError < bestGoalSP) isBestSP = WORSE;
    else isBestSP = EQUAL;
  }
};

void PatternRecognition::showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError)
{
  if (useSP) {REPORT("Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << " SP (val) = " << valError)}
  else Training::showTrainingStatus(epoch, trnError, valError);
};

