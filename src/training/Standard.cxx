#include "fastnet/training/Standard.h"

StandardTraining::StandardTraining(FastNet::Backpropagation *net, DataManager *inTrn, DataManager *outTrn, DataManager *inVal, mxArray *outVal, const unsigned bSize) : Training(net, bSize)
{
  DEBUG2("Creating StandardTraining object.");
  
  inTrnData = inTrn;
  outTrnData = outTrn;
  inValData = inVal;
  outValData = outVal;
};

void StandardTraining::valNetwork(REAL &mseVal, REAL &spVal)
{
  REAL gbError = 0.;
  REAL error = 0.;
  const REAL *output;

  const DataManager *input = inValData;
  const DataManager *target = outValData;
  const int numEvents = static_cast<int>(inValData->numEvents());
  
  int chunk = chunkSize;
  int i, thId;
  FastNet::Backpropagation **nv = netVec;

  #pragma omp parallel shared(input,target,chunk,nv,gbError) private(i,thId,output,error)
  {
    thId = omp_get_thread_num();
    error = 0.;

    #pragma omp for schedule(dynamic,chunk) nowait
    for (i=0; i<numEvents; i++)
    {
      error += nv[thId]->applySupervisedInput((*input)[i], (*target)[i], output);
    }

    #pragma omp critical
    gbError += error;
  }
  
  mseVal = gbError / static_cast<REAL>(numEvents);
};


REAL StandardTraining::trainNetwork()
{
  unsigned pos;
  REAL gbError = 0.;
  REAL error = 0.;
  const REAL *output;

  DataManager *input = inTrnData;
  const DataManager *target = outTrnData;

  int chunk = chunkSize;
  int i, thId;
  FastNet::Backpropagation **nv = netVec;
  const int nEvents = (batchSize) ? batchSize : dm->numEvents();

  #pragma omp parallel shared(input,target,chunk,nv,gbError) private(i,thId,output,error,pos)
  {
    thId = omp_get_thread_num(); 
    error = 0.;

    #pragma omp for schedule(dynamic,chunk) nowait
    for (i=0; i<nEvents; i++)
    {
        #pragma omp critical
        pos = input->getNextEventIndex();
        
        error += nv[thId]->applySupervisedInput((*input)[pos], (*target)[pos], output);
        nv[thId]->calculateNewWeights(output, (*target)[pos]);
    }

    #pragma omp critical
    gbError += error;    
  }

  updateGradients();
  updateWeights();
  return (gbError / static_cast<REAL>(nEvents));
}

  
void StandardTraining::showInfo(const unsigned nEpochs) const
{
  REPORT("TRAINING DATA INFORMATION (Standard Network)");
  REPORT("Number of Epochs          : " << nEpochs);
};
