#include "fastnet/training/Standard.h"

StandardTraining::StandardTraining(const mxArray *inTrn, const mxArray *outTrn, const mxArray *inVal, const mxArray *outVal) : Training()
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


REAL StandardTraining::valNetwork(FastNet::Backpropagation *net)
{
  REAL gbError = 0.;
  const REAL *out;

  const REAL *input = inValData;
  const REAL *target = outValData;
  for (unsigned i=0; i<numValEvents; i++)
  {
    gbError += net->applySupervisedInput(input, target, out);
    input += inputSize;
    target += outputSize;
  }
  return (gbError / static_cast<REAL>(numValEvents));
};


REAL StandardTraining::trainNetwork(FastNet::Backpropagation *net)
{
  unsigned evIndex;
  REAL gbError = 0.;
  const REAL *output;

  const REAL *input = inTrnData;
  const REAL *target = outTrnData;
  for (unsigned i=0; i<numTrnEvents; i++)
  {
    gbError += net->applySupervisedInput(input, target, output);
    net->calculateNewWeights(output, target);
    input += inputSize;
    target += outputSize;
  }
  return (gbError / static_cast<REAL>(numTrnEvents));
}
  
void StandardTraining::checkSizeMismatch(const FastNet::Backpropagation *net) const
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
