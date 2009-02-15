/** 
@file  nsim.cpp
@brief The Matlab's nsim function definition file.

 This file implements the function that is called by matlab when the matlab's nsim function
 is called. This function reads the matlab arguments (specified in "args"), and porpagates
 the input data set through the passed neural  network, returning the outputs obtained.
*/

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mex.h>

#include "fastnet/reporter/Reporter.h"
#include "fastnet/neuralnet/feedforward.h"
#include "fastnet/defines.h"

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

const unsigned NUM_THREADS = 2;

struct ThreadParams
{
  REAL *inputStartAddr;  // Input starting address to start processing.
  REAL *outputStartAddr;  // Output starting address to start processing.
  unsigned numEvents; // Total number of events to process.
  FeedForward *net; // The neural network to use.
};


void *threadRun(void *params)
{
  ThreadParams *par = static_cast<ThreadParams*>(params);
  FeedForward *net = par->net;
  const unsigned inputSize = (*net)[0];
  const unsigned outputSize = (*net)[par->net->getNumLayers()-1];
  const unsigned numBytes2Copy = outputSize * sizeof(REAL);
  for (unsigned i=0; i<par->numEvents; i++)
  {
    memcpy(par->outputStartAddr, par->net->propagateInput(par->inputStartAddr), numBytes2Copy);
    par->inputStartAddr += inputSize;
    par->outputStartAddr += outputSize;
  } 
  pthread_exit(NULL);
}

/// Matlab 's main function.
void mexFunction(int nargout, mxArray *ret[], int nargin, const mxArray *args[])
{
  try
  {  
    //Verifying if the number of input parameters is ok.
    if (nargin != NUM_ARGS) throw "Incorrect number of arguments! See help for information!";

    //Reading the configuration structure
    const mxArray *netStr = args[NET_STR_IDX];
    
    // Creating the neural network to use.
    FeedForward net(netStr);

    //Checking if the input and output data sizes match the network's input layer.
    if (mxGetM(args[IN_DATA_IDX]) != net[0])
      throw "Input training or testing data do not match the network input layer size!";

    //Creating the input and output access matrices.
    const unsigned numEvents = mxGetN(args[IN_DATA_IDX]);
    const unsigned inputSize = mxGetM(args[IN_DATA_IDX]);
    const unsigned outputSize = net[net.getNumLayers()-1];
    REAL *inputEvents = static_cast<REAL*>(mxGetData(args[IN_DATA_IDX]));
    mxArray *outData = mxCreateNumericMatrix(outputSize, numEvents, REAL_TYPE, mxREAL);
    REAL *outputEvents = static_cast<REAL*>(mxGetData(outData));

    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;
    //Setando as threads p/ serem joinable.
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    ThreadParams *params = new ThreadParams[NUM_THREADS];
    const unsigned evPerThread = numEvents / NUM_THREADS;
    for (unsigned i=0; i<NUM_THREADS; i++)
    {
      params[i].net = new FeedForward(net);
      //The first thread takes the remaining elements, in case of odd division.
      params[i].numEvents = (i) ? evPerThread : (evPerThread + (numEvents % NUM_THREADS));
      params[i].inputStartAddr = inputEvents;
      params[i].outputStartAddr = outputEvents;
      inputEvents += (params[i].numEvents * inputSize);
      outputEvents += (params[i].numEvents * outputSize);
      const int rc = pthread_create(&threads[i], &attr, threadRun, (void *)&params[i]);
      if (rc) throw "Impossible to create thread! Aborting...";
    }
    
    //Free attribute and wait for the other threads
    pthread_attr_destroy(&attr);
    for (unsigned i=0; i<NUM_THREADS; i++)
    {
      void *ret;
      pthread_join(threads[i], &ret);
      delete params[i].net;
    }
    
    ret[NET_OUT_IDX] = outData;
  }
  catch (const char *msg) FATAL(msg);
}
