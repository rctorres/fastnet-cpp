/** 
@file  backpropagation.cpp
@brief The BackPropagation class definition.
*/

#include <vector>
#include <string>

#include "fastnet/neuralnet/backpropagation.h"

namespace FastNet
{
	Backpropagation::Backpropagation(vector<unsigned>nodesDist, vector<string> trfFunction, REAL lrnRate, REAL decFac) : NeuralNetwork(nodesDist, trfFunction)
	{
		const unsigned size = nLayers - 1;

		//Initializing the pointers.
		sigma = NULL;
		dw = NULL;
		db = NULL;

		//Initializing class attributes.
		learningRate = lrnRate;
		decFactor = decFac;
		trnEventCounter = 0;
		
		try
		{
			//Allocating the delta bias matrix.
			db = new REAL* [size];
			//Initiallizing with NULL in case of an future error on allocating memory.
			for (unsigned i=0; i<size; i++) db[i] = NULL;
			//Allocating the matrix's collumns.
			for (unsigned i=0; i<size; i++)
			{
				db[i] = new REAL [nNodes[i+1]];
				for (unsigned j=0; j<nNodes[i+1]; j++) db[i][j] = 0;
			}

			//Allocating the delta weight matrix.
			dw = new REAL** [size];
			for (unsigned i=0; i<size; i++) dw[i] = NULL;
			for (unsigned i=0; i<size; i++)
			{
				dw[i] = new REAL* [nNodes[i+1]];
				for (unsigned j=0; j<nNodes[i+1]; j++) dw[i][j] = NULL;
				for (unsigned j=0; j<nNodes[i+1]; j++) 
				{
					dw[i][j] = new REAL [nNodes[i]];
					for (unsigned k=0; k<nNodes[i]; k++) dw[i][j][k] = 0;
				}
			}

			//Allocating space for the sigma matrix.
			sigma = new REAL* [size];
			for (unsigned i=0; i<size; i++) sigma[i] = NULL;
			for (unsigned i=0; i<size; i++) sigma[i] = new REAL [nNodes[i+1]];
		}
		catch (bad_alloc xa)
		{
			throw;
		}
	}



	Backpropagation::~Backpropagation()
	{
		releaseMatrix(db);
		releaseMatrix(dw);
		releaseMatrix(sigma);
	}

	void Backpropagation::retropropagateError(const REAL *output, const REAL *target)
	{
		const unsigned size = nLayers - 1;

		for (unsigned i=activeNodes[size].init; i<activeNodes[size].end; i++) sigma[size-1][i] = (target[i] - output[i]) * CALL_TRF_FUNC(trfFunc[size-1])(output[i], true);

		//Retropropagating the error.
		for (int i=(size-2); i>=0; i--)
		{
			for (unsigned j=activeNodes[i+1].init; j<activeNodes[i+1].end; j++)
			{
				sigma[i][j] = 0;

				for (unsigned k=activeNodes[i+2].init; k<activeNodes[i+2].end; k++)
				{
					sigma[i][j] += sigma[i+1][k] * weights[(i+1)][k][j];
				}

				sigma[i][j] *= CALL_TRF_FUNC(trfFunc[i])(layerOutputs[i+1][j], true);
			}
		}
	}
	

	void Backpropagation::calculateNewWeights(const REAL *output, const REAL *target)
	{
    const unsigned size = nLayers - 1;

		retropropagateError(output, target);

		//Accumulating the deltas.
		for (unsigned i=0; i<size; i++)
		{
			for (unsigned j=activeNodes[(i+1)].init; j<activeNodes[(i+1)].end; j++)
			{
				for (unsigned k=activeNodes[i].init; k<activeNodes[i].end; k++)
				{
					dw[i][j][k] += (sigma[i][j] * layerOutputs[i][k]);
				}

				db[i][j] += sigma[i][j];
			}
		}

		//Increasing the event counter, for batch training.
		trnEventCounter++;
	}

	void Backpropagation::calculateNewWeights(const REAL *output, const REAL *target, unsigned nEv, unsigned nPat)
	{
		const unsigned size = nLayers - 1;
		const REAL val = 1.0 / static_cast<REAL>(nEv * nPat);

		retropropagateError(output, target);

		//Accumulating the deltas.
		for (unsigned i=0; i<size; i++)
		{
			for (unsigned j=activeNodes[(i+1)].init; j<activeNodes[(i+1)].end; j++)
			{
				for (unsigned k=activeNodes[i].init; k<activeNodes[i].end; k++)
				{
					dw[i][j][k] += (val * sigma[i][j] * layerOutputs[i][k]);
				}

				db[i][j] += (val * sigma[i][j]);
			}
		}

		// We will not use the trnEventCounter in this case.
		trnEventCounter = 1;
	}
	
	void Backpropagation::updateWeights()
	{		
		//If the new weights has not been calculated, the function is aborted.
		if (!trnEventCounter) return;

		// Using the inverse, in order to improve speed.
		const REAL invNTrnEv = 1 / static_cast<REAL>(trnEventCounter);

		for (unsigned i=0; i<(nLayers-1); i++)
		{
			for (unsigned j=activeNodes[i+1].init; j<activeNodes[(i+1)].end; j++)
			{
				//If the node is frozen, we just reset the accumulators,
				//otherwise, we actually train the weights connected to it.
				if (frozenNode[i][j])
				{
					for (unsigned k=activeNodes[i].init; k<activeNodes[i].end; k++) dw[i][j][k] = 0;
					db[i][j] = 0;
					if (!usingBias[i]) bias[i][j] = 0;
				}
				else
				{
					for (unsigned k=activeNodes[i].init; k<activeNodes[i].end; k++)
					{
						weights[i][j][k] += (learningRate * (dw[i][j][k] * invNTrnEv));
						dw[i][j][k] = 0;
					}

					if (usingBias[i])
					{
						bias[i][j] += (learningRate * (db[i][j] * invNTrnEv));
						db[i][j] = 0;
					}
					else
					{
						bias[i][j] = 0;
					}
				}
			}
		}

		trnEventCounter = 0;
	}
	
	void Backpropagation::showInfo(ostream &str) const
	{
		NeuralNetwork::showInfo(str);
		str << "TRAINING ALGORITHM INFORMATION:" << endl;
		str << "Training algorithm : Gradient Descent" << endl;
		str << "Learning rate      : " << learningRate << endl;
		str << "Decreasing factor  : " << decFactor << endl;
	}
}