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

#include "TrigRingerTools/matlab/MatlabReporter.h"
#include "TrigRingerTools/fastnet/backpropagation.h"
#include "TrigRingerTools/fastnet/rprop.h"
#include "TrigRingerTools/matlab/matevents.h"
#include "TrigRingerTools/matlab/matnetdata.h"
#include "TrigRingerTools/fastnet/defines.h"
#include "TrigRingerTools/matlab/mxhandler.h"

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
  sys::Reporter *reporter = new sys::MatlabReporter();

	try
	{	
		//Verifying if the number of input parameters is ok.
		if (nargin != NUM_ARGS) throw "Incorrect number of arguments! See help for information!";

		//Reading the configuration structure
		const mxArray *netStr = args[NET_STR_IDX];
		
    MatEvents inputData(args[IN_DATA_IDX]);
		
		vector<unsigned> nNodes;
		//Getting the number of nodes in the input layer.
		nNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(mxGetCell(mxGetField(netStr, 0, "inputs"), 0), 0, "size"))));
		
		//Getting the number of nodes and transfer function in each hidden layer:
		const mxArray *layers = mxGetField(netStr, 0, "layers");
		vector<string> trfFunc;
		for (size_t i=0; i<mxGetM(layers); i++)
		{
			mxArray *layer = mxGetCell(layers, i);
			nNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(layer, 0, "size"))));
			string transFunction = mxArrayToString(mxGetField(layer, 0, "transferFcn"));
			trfFunc.push_back(transFunction);
		}

		//Checking if the input and output data sizes match the network's input layer.
		if (mxGetM(args[IN_DATA_IDX]) != nNodes[0])
		  throw "Input training or testing data do not match the network input layer size!";

    // Creating the neural network to use. It can be any type, since we will just propagate the
    // inputs.
    RProp net(nNodes, trfFunc);
		
		//Creating the network data handler.
		MatNetData netData(nNodes, netStr);
		
		//Initializing the weights and biases.
		net.readWeights(&netData);
		
		//Getting the active nodes of the input layer.
		const mxArray *userData = mxGetField(mxGetCell(mxGetField(netStr, 0, "inputs"), 0), 0, "userdata");
		const mxArray *initNode = mxGetField(userData, 0, "initNode");
		const mxArray *endNode = mxGetField(userData, 0, "endNode");
		if ( (initNode) && (endNode) )
		{
			const unsigned init = static_cast<unsigned>(mxGetScalar(initNode)) - 1;
			const unsigned end = static_cast<unsigned>(mxGetScalar(endNode)) - 1;
			if ( (init <= end) && (end < nNodes[0]) ) net.setActiveNodes(0, init, end);
			else throw "Invalid nodes init or end values!";
		}
		
		// This loop also set if a given layer is not using bias and the start and end
		//nodes of each layer.
		for (unsigned i=0; i<mxGetM(layers); i++)
		{
			//Getting the nodes range information.
			userData = mxGetField(mxGetCell(layers, i), 0, "userdata");
			initNode = mxGetField(userData, 0, "initNode");
			endNode = mxGetField(userData, 0, "endNode");
			if ( (initNode) && (endNode) )
			{
				const unsigned init = static_cast<unsigned>(mxGetScalar(initNode)) - 1;
				const unsigned end = static_cast<unsigned>(mxGetScalar(endNode)) - 1;
				if ( (init <= end) && (end < nNodes[(i+1)]) ) net.setActiveNodes((i+1), init, end);
				else throw "Invalid nodes init or end values!";
			}
			
			//Getting the using bias information.
			const mxArray *usingBias = mxGetField(userData, 0, "usingBias");
			if (usingBias) net.setUsingBias(i, static_cast<bool>(mxGetScalar(usingBias)));			
		}		

    //Creating the output matrix.
    const unsigned outputSize = nNodes[nNodes.size()-1];
    const unsigned bytes2Copy = outputSize*sizeof(double);
    mxArray *outData = mxCreateDoubleMatrix(outputSize, inputData.getNumEvents(), mxREAL);
    double *outPtr = mxGetPr(outData);

    while  (inputData.hasNext())
    {
      memcpy(outPtr, net.propagateInput(inputData.readEvent()), bytes2Copy);
      outPtr += outputSize;
    }
    
    ret[NET_OUT_IDX] = outData;
	}
	catch (const char *msg) RINGER_FATAL(reporter, msg);
	
	delete reporter;
}
