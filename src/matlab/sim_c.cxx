/** 
@file  nsim.cpp
@brief The Matlab's nsim function definition file.

 This file implements the function that is called by matlab when the matlab's nsim function
 is called. This function reads the matlab arguments (specified in "args"), and porpagates
 the input data set through the passed neural  network, returning the outputs obtained.
*/

#include <mex.h>
#include <omp.h>

#include "fastnet/sys/Reporter.h"
#include "fastnet/neuralnet/feedforward.h"
#include "fastnet/sys/defines.h"

using namespace std;
using namespace FastNet;

/// Number of input arguments.
const unsigned NUM_ARGS = 2;

/// Index, in the arguments list, of the neural network structure.
const unsigned NET_STR_IDX = 0;

/// Index, in the arguments list, of the input testing events.
const unsigned IN_DATA_IDX = 1;

/// Index, in the return vector, of the network structure after training.
const unsigned NET_OUT_IDX = 0;


/// Matlab 's main function.
void mexFunction(int nargout, mxArray *ret[], int nargin, const mxArray *args[])
{
  try
  {
    //Verifying if the number of input parameters is ok.
    if (nargin != NUM_ARGS) throw "Incorrect number of arguments! See help for information!";

    // Creating the neural network to use.
    FeedForward net(args[NET_STR_IDX]);

    //Checking if the input and output data sizes match the network's input layer.
    if (mxGetM(args[IN_DATA_IDX]) != net[0])
      throw "Input training or testing data do not match the network input layer size!";

    //Creating the input and output access matrices.
    const unsigned numEvents = mxGetN(args[IN_DATA_IDX]);
    const unsigned inputSize = mxGetM(args[IN_DATA_IDX]);
    const unsigned outputSize = net[net.getNumLayers()-1];
    const unsigned numBytes2Copy = outputSize * sizeof(REAL);
    REAL *inputEvents = static_cast<REAL*>(mxGetData(args[IN_DATA_IDX]));
    mxArray *outData = mxCreateNumericMatrix(outputSize, numEvents, REAL_TYPE, mxREAL);
    REAL *outputEvents = static_cast<REAL*>(mxGetData(outData));
    
    int i;
    int chunk = 1000;
    #pragma omp parallel shared(inputEvents,outputEvents,chunk) private(i) firstprivate(net)
    {
      #pragma omp for schedule(dynamic,chunk) nowait
      for (i=0; i<numEvents; i++)
      {
        memcpy(&outputEvents[i*outputSize], net.propagateInput(&inputEvents[i*inputSize]), numBytes2Copy);
      }
    }

    ret[NET_OUT_IDX] = outData;
  }
  catch (const char *msg) FATAL(msg);
}
