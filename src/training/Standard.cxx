#include "fastnet/training/Standard.h"

StandardTraining::StandardTraining(FastNet::Backpropagation *net, const mxArray *inTrn, const mxArray *outTrn, const mxArray *inVal, const mxArray *outVal) : Training(net)
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
  numTrnEvents = static_cast<unsigned>(mxGetN(inTrn));
  numValEvents = static_cast<unsigned>(mxGetN(inVal));
  inputSize = static_cast<unsigned>(mxGetM(inTrn));
  outputSize = static_cast<unsigned>(mxGetM(outTrn));
};


REAL StandardTraining::valNetwork()
{
  REAL gbError = 0.;
  REAL error = 0.;
  const REAL *out;

  const REAL *input = inValData;
  const REAL *target = outValData;
  
  int chunk = 1000;
  int i, thId;
  FastNet::Backpropagation **nv = netVec;

  #pragma omp parallel shared(input,target,chunk,nv,gbError) private(i,thId,out,error)
  {
    thId = omp_get_thread_num();
    error = 0.;

    #pragma omp for schedule(dynamic,chunk) nowait
    for (i=0; i<numValEvents; i++)
    {
      error += nv[thId]->applySupervisedInput(&input[i*inputSize], &target[i*outputSize], out);
    }

    #pragma omp atomic
    gbError += error;
  }
  return (gbError / static_cast<REAL>(numValEvents));
};


REAL StandardTraining::trainNetwork()
{
  unsigned evIndex;
  REAL gbError = 0.;
  REAL error = 0.;
  const REAL *output;

  const REAL *input = inTrnData;
  const REAL *target = outTrnData;

  int chunk = 1000;
  int i, thId;
  FastNet::Backpropagation **nv = netVec;

  updateNetworks();
  #pragma omp parallel shared(input,target,chunk,nv,gbError) private(i,thId,output,error)
  {
    thId = omp_get_thread_num(); 
    error = 0.;

    #pragma omp for schedule(dynamic,chunk) nowait
    for (i=0; i<numTrnEvents; i++)
    {
        error += nv[thId]->applySupervisedInput(&input[i*inputSize], &target[i*outputSize], output);
        nv[thId]->calculateNewWeights(output, &target[i*outputSize]);
    }

    #pragma omp atomic
    gbError += error;    
  }

  updateGradients();
  return (gbError / static_cast<REAL>(numTrnEvents));
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
  REPORT("Total number of training events   : " << numTrnEvents);
  REPORT("Total number of validating events    : " << numValEvents);
};
