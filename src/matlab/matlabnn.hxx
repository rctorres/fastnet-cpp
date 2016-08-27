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

class MatlabNN : public NeuralNetwork
{
    public:
        static void get_mx_parameters(const mxArray *netStr, vector<unsigned> &mat_numNodes, vector<string> &mat_trfFunc, vector<bool> &mat_usingBias)
        {
            DEBUG1("Clearing output vectors..");
            mat_numNodes.clear();
            mat_trfFunc.clear();
            mat_usingBias.clear();
            
            DEBUG1("Getting the constructor parameters from the Matlab structure.");
            //Getting the number of nodes in the input layer.
            mat_numNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(mxGetCell(mxGetField(netStr, 0, "inputs"), 0), 0, "size"))));
    
            //Getting the number of nodes and transfer function in each layer:
            const mxArray *layers = mxGetField(netStr, 0, "layers");
            for (size_t i=0; i<mxGetM(layers); i++)
            {
                const mxArray *layer = mxGetCell(layers, i);
                //Getting layer size
                mat_numNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(layer, 0, "size"))));
                //Getting transfer function
                mat_trfFunc.push_back(mxArrayToString(mxGetField(layer, 0, "transferFcn")));
                    
                //Getting the using bias information.
                const mxArray *userData = mxGetField(mxGetCell(layers, i), 0, "userdata");
                mat_usingBias.push_back(static_cast<bool>(mxGetScalar(mxGetField(userData, 0, "usingBias"))));
            }         
        }
    
        MatlabNN(const mxArray *netStr, const vector<unsigned> &mat_numNodes,
                       const vector<string> &mat_trfFunc, const vector<bool> &mat_usingBias) : NeuralNetwork(mat_numNodes, mat_trfFunc, mat_usingBias)
        {
            DEBUG1("Initializing the NeuralNetwork class from a Matlab Network structure.");
            //Taking the weights and values info.
            readWeights(netStr);
        }

        void readWeights(const mxArray *mNet)
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

            for (unsigned i=0; i<nNodes[1]; i++)
            {
                for (unsigned j=0; j<nNodes[0]; j++)
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
    
            for (unsigned i=1; i<(nNodes.size()-1); i++)
            {
                iw = mxGetCell(lw, iw.getPos(i,(i-1), mxGetM(lw)));
                ib = mxGetCell(lb, i);

                for (unsigned j=0; j<nNodes[(i+1)]; j++)
                {
                    for (unsigned k=0; k<nNodes[i]; k++)
                    {
                        weights[i][j][k] = static_cast<REAL>(iw(j,k));
                        DEBUG3("Weight[" << i << "][" << j << "][" << k << "] = " << weights[i][j][k]);
                    }
                    bias[i][j] = (usingBias[i]) ? static_cast<REAL>(ib(j)) : 0.;
                    DEBUG3("Bias[" << i << "][" << j << "] = " << bias[i][j]);
                }
            }
        }
        
        MatlabNN(const MatlabNN &other) : NeuralNetwork(other){}
        virtual NeuralNetwork *clone() {return new MatlabNN(*this);}
};
#endif
