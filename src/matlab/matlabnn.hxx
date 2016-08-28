/** 
@file  feedforward.h
@brief A simple feedforward network class.
*/

 
#ifndef MATLABNN_H
#define MATLABNN_H

#include <vector>
#include <cstring>

#include <mex.h>

#include "fastnet/sys/defines.h"
#include "fastnet/neuralnet/neuralnetwork.h"
#include "mxhandler.hxx"

/** 
@brief    Binds fastnet to a Matlab Net class.
@author    Rodrigo Coura Torres (torres@lps.ufrj.br)
@version  1.0
@date    23/01/2009

This class should be used for network production, when no training is necessary,
just feedforward the incoming events, fot output collection.
*/

using namespace FastNet;
using namespace std;

class MatlabNN
{
    protected:
        vector<unsigned> numNodes;
        vector<string> trfFunc;
        vector<bool> usingBias;
        REAL ***weights;
        REAL **bias;

        void allocate_space()
        {
            try
              {
                  const unsigned size = numNodes.size()-1;
                  bias = new REAL* [size];
                  weights = new REAL** [size];

                  for (unsigned i=0; i<size; i++)
                  {
                      bias[i] = new REAL [numNodes[i+1]];
                      weights[i] = new REAL* [numNodes[i+1]];
                      for (unsigned j=0; j<numNodes[i+1]; j++) weights[i][j] = new REAL [numNodes[i]];
                  }
              }
              catch (bad_alloc xa)
              {
                  throw;
              }
        }

        virtual void readWeights(const mxArray *mNet)
        {
            // It must be of double tye, since the matlab net tructure holds its info with
            //double precision.
            MxArrayHandler<double> iw, ib;
            mxArray *lw;
            mxArray *lb;

            //Getting the bias cells vector.
            lb = mxGetField(mNet, 0, "b");

            //Processing first the input layer.
            iw = mxGetCell(mxGetField(mNet, 0, "IW"), 0);
            ib = mxGetCell(lb, 0);

            for (unsigned i=0; i<numNodes[1]; i++)
            {
                for (unsigned j=0; j<numNodes[0]; j++)
                {
                    weights[0][i][j] = static_cast<REAL>(iw(i,j));
                    DEBUG3("Weight[0][" << i << "][" << j << "] = " << weights[0][i][j]);
                }
                bias[0][i] = (usingBias[0]) ? static_cast<REAL>(ib(i)) : 0.;
                DEBUG3("Bias[0][" << i << "] = " << bias[0][i]);
            }
    
            //Processing the other layers.
            //Getting the weights cell matrix.
            lw = mxGetField(mNet, 0, "LW");
    
            for (unsigned i=1; i<(numNodes.size()-1); i++)
            {
                iw = mxGetCell(lw, iw.getPos(i,(i-1), mxGetM(lw)));
                ib = mxGetCell(lb, i);

                for (unsigned j=0; j<numNodes[(i+1)]; j++)
                {
                    for (unsigned k=0; k<numNodes[i]; k++)
                    {
                        weights[i][j][k] = static_cast<REAL>(iw(j,k));
                        DEBUG3("Weight[" << i << "][" << j << "][" << k << "] = " << weights[i][j][k]);
                    }
                    bias[i][j] = (usingBias[i]) ? static_cast<REAL>(ib(j)) : 0.;
                    DEBUG3("Bias[" << i << "][" << j << "] = " << bias[i][j]);
                }
            }
        }

    public:
        MatlabNN(const mxArray *net)
        {
            DEBUG1("Initializing the NeuralNetwork class from a Matlab Network structure.");
            weights = NULL;
            bias = NULL;

            DEBUG1("Getting the constructor parameters from the Matlab structure.");
            //Getting the number of nodes in the input layer.
            numNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(mxGetCell(mxGetField(net, 0, "inputs"), 0), 0, "size"))));

            //Getting the number of nodes and transfer function in each layer:
            const mxArray *layers = mxGetField(net, 0, "layers");
            for (size_t i=0; i<mxGetM(layers); i++)
            {
                const mxArray *layer = mxGetCell(layers, i);
                //Getting layer size
                numNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(layer, 0, "size"))));
                //Getting transfer function
                trfFunc.push_back(mxArrayToString(mxGetField(layer, 0, "transferFcn")));

                //Getting the using bias information.
                const mxArray *userData = mxGetField(mxGetCell(layers, i), 0, "userdata");
                usingBias.push_back(static_cast<bool>(mxGetScalar(mxGetField(userData, 0, "usingBias"))));
            } 

            //Taking the weights and values info.
            allocate_space();
            readWeights(net);
        }

        virtual NeuralNetwork *getNetwork()
        {
            NeuralNetwork *ret = new NeuralNetwork(numNodes, trfFunc, usingBias);
            ret->readWeights( (const REAL***) weights, (const REAL**) bias);
            return ret;
        }

        virtual ~MatlabNN()
        {
            if (weights)
            {
                DEBUG1("Releasing allocated memory for weights.");
                for (int i=0; i<(numNodes.size()-1); i++)
                {
                    if (weights[i])
                    {
                        for (int j=0; j<numNodes[i+1]; j++)
                        {
                            if (weights[i][j]) delete [] weights[i][j];
                        }
                        delete [] weights[i];
                    }
                }
                delete [] weights;
                weights = NULL;
            }
          
            if (bias)
            {
                DEBUG1("Releasing allocated memory for biases.");
                for (int i=0; i<(numNodes.size()-1); i++)
                {
                    if (bias[i]) delete [] bias[i];
                }
                delete [] bias;
                bias = NULL;
            }
        }
};
#endif
