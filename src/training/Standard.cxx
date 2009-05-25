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
  numValEvents = static_cast<int>(mxGetN(inVal));
};

StandardTraining::~StandardTraining()
{
  delete dmTrn;
}

REAL StandardTraining::valNetwork()
{
  REAL gbError = 0.;
  REAL error = 0.;
  const REAL *output;

  const REAL *input = inValData;
  const REAL *target = outValData;
  
  int chunk = chunkSize;
  int i, thId;
  FastNet::Backpropagation **nv = netVec;

  #pragma omp parallel shared(input,target,chunk,nv,gbError,numValEvents) private(i,thId,output,error)
  {
    thId = omp_get_thread_num();
    error = 0.;

    #pragma omp for schedule(dynamic,chunk) nowait
    for (i=0; i<numValEvents; i++)
    {
      error += nv[thId]->applySupervisedInput(&input[i*inputSize], &target[i*outputSize], output);
    }

    #pragma omp critical
    gbError += error;
  }
  return (gbError / static_cast<REAL>(numValEvents));
};


REAL StandardTraining::trainNetwork()
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

  updateNetworks();
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
  return (gbError / static_cast<REAL>(batchSize));
}
  
void StandardTraining::checkSizeMismatch() const
{
  if (inputSize != (*net)[0])
    throw "Input training or validating data do not match the network input layer size!";

  if ( outputSize != (*net)[net->getNumLayers()-1] )
    throw "Output training or validating data do not match the network output layer size!";
};
  
void StandardTraining::showInfo(const unsigned nEpochs) const
{
  REPORT("TRAINING DATA INFORMATION (Standard Network)");
  REPORT("Number of Epochs          : " << nEpochs);
};
