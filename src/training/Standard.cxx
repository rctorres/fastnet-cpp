#include "fastnet/training/Standard.h"

StandardTraining::StandardTraining(FastNet::Backpropagation *net, const mxArray *inTrn, const mxArray *outTrn, const mxArray *inVal, const mxArray *outVal, const unsigned bSize) : Training(net, bSize)
{
  DEBUG2("Creating StandardTraining object.");
  
  if ( mxGetM(inTrn) != mxGetM(inVal) ) throw "Input training and validating events dimension does not match!";
  if ( mxGetM(outTrn) != mxGetM(outVal) ) throw "Output training and validating events dimension does not match!";
  if ( mxGetN(inTrn) != mxGetN(outTrn) ) throw "Number of input and target training events does not match!";
  if ( mxGetN(inVal) != mxGetN(outVal) ) throw "Number of input and target validating events does not match!";

  inTrnData = static_cast<REAL*>(mxGetData(inTrn));
  outTrnData = static_cast<REAL*>(mxGetData(outTrn));
  inValData = static_cast<REAL*>(mxGetData(inVal));
  outValData = static_cast<REAL*>(mxGetData(outVal));
  inputSize = static_cast<unsigned>(mxGetM(inTrn));
  outputSize = static_cast<unsigned>(mxGetM(outTrn));
  
  dmTrn = new DataManager(static_cast<unsigned>(mxGetN(inTrn)));
  numValEvents = static_cast<unsigned>(mxGetN(inVal));
};

StandardTraining::~StandardTraining()
{
  delete dmTrn;
}

void StandardTraining::valNetwork(REAL &mseVal, REAL &spVal)
{
  REAL gbError = 0.;
  REAL error = 0.;
  const REAL *output;

  const REAL *input = inValData;
  const REAL *target = outValData;
  const int numEvents = static_cast<int>(numValEvents);
  
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
      error += nv[thId]->applySupervisedInput(&input[i*inputSize], &target[i*outputSize], output);
    }

    #pragma omp critical
    gbError += error;
  }
  
  mseVal = gbError / static_cast<REAL>(numEvents);
};


REAL StandardTraining::trainNetwork(const bool firstEpoch)
{
  unsigned pos;
  REAL gbError = 0.;
  REAL error = 0.;
  const REAL *output;

  const REAL *input = inTrnData;
  const REAL *target = outTrnData;

  int chunk = chunkSize;
  int i, thId;
  FastNet::Backpropagation **nv = netVec;
  DataManager *dm = dmTrn;

  #pragma omp parallel shared(input,target,chunk,nv,gbError,dm) private(i,thId,output,error,pos)
  {
    thId = omp_get_thread_num(); 
    error = 0.;

    #pragma omp for schedule(dynamic,chunk) nowait
    for (i=0; i<batchSize; i++)
    {
        #pragma omp critical
        pos = dm->get();
        
        error += nv[thId]->applySupervisedInput(&input[pos*inputSize], &target[pos*outputSize], output);
        nv[thId]->calculateNewWeights(output, &target[pos*outputSize]);
    }

    #pragma omp critical
    gbError += error;    
  }

  updateGradients();
  updateWeights();
  return (gbError / static_cast<REAL>(batchSize));
}

  
void StandardTraining::showInfo(const unsigned nEpochs) const
{
  REPORT("TRAINING DATA INFORMATION (Standard Network)");
  REPORT("Number of Epochs          : " << nEpochs);
};
