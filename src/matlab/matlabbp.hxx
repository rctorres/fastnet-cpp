/** 
@file  feedforward.h
@brief A class to bind the backpropagation class to Matlab.
*/

 
#ifndef MATLABBP_H
#define MATLABBP_H

#include "fastnet/sys/defines.h"
#include "fastnet/neuralnet/neuralnetwork.h"
#include "matlabnn.hxx"

using namespace std;

/** 
@brief    Binds fastnet BackPropagation to a Matlab Net class.
@author    Rodrigo Coura Torres (torres@lps.ufrj.br)
@version  1.0
@date    23/01/2009

Used for setting up a Backpropagation network based on a Matlab NN object.
*/
class MatlabBP : public MatlabNN 
{
    protected:
        struct Node {unsigned layer, node;};

        REAL learningRate;
        REAL decFactor;
        list<Node> frozen;
    
    public:
        MatlabBP(const mxArray *netStr, const mxArray *trnParam) : MatlabNN(netStr)
        {
            DEBUG1("Initializing the Backpropagation class from a Matlab Network structure.");

            //We first test whether the values exists, otherwise, we use default ones.
            if (mxGetField(trnParam, 0, "lr")) this->learningRate = static_cast<REAL>(abs(mxGetScalar(mxGetField(trnParam, 0, "lr"))));
            else this->learningRate = 0.05;
            if (mxGetField(trnParam, 0, "decFactor")) this->decFactor = static_cast<REAL>(abs(mxGetScalar(mxGetField(trnParam, 0, "decFactor"))));
            else this->decFactor = 1;


            //Verifying if there are frozen nodes and seting them.
            const mxArray *layers = mxGetField(netStr, 0, "layers");
            for (unsigned i=0; i<(numNodes.size()-1); i++)
            {
                //Getting from Matlab any possible frozen nodes.
                const mxArray *userData = mxGetField(mxGetCell(layers, i), 0, "userdata");
                const mxArray *matFNodes = mxGetField(userData, 0, "frozenNodes");
                const double *fNodes = mxGetPr(matFNodes);
                for (unsigned j=0; j<mxGetN(matFNodes); j++)
                {
                    const unsigned node = static_cast<unsigned>(fNodes[j]) - 1;
                    if (node < numNodes[(i+1)]) frozen.push_back({.layer = i, .node = node});
                    else throw "Node to be frozen is invalid!";
                }
            }
        }

        virtual Backpropagation *getNetwork()
        {
            Backpropagation *ret = new Backpropagation(numNodes, trfFunc, usingBias, learningRate, decFactor);
            ret->readWeights( (const REAL***) weights, (const REAL**) bias);
            for (list<Node>::const_iterator itr = frozen.begin(); itr != frozen.end(); itr++) ret->setFrozen(itr->layer, itr->node, true);
            return ret;
        }

        /// Flush weights from memory to a Matlab variable.
        /**
        Since this class, in order to optimize speed, saves the
        weights and bias values into memory, at the end, if the user wants
        to save the final values, this method must be called. It will
        save the weights and biases values stored in the memory buffer in a matlab variable.
        So, this method can only be used after the writeWeights has been called at least once.
        @param[out] outNet The matlab network structure to where the weights and biases will be saved to.
        */
        virtual void flushBestTrainWeights(mxArray *outNet, const Backpropagation *net) const
        {
            // It must be of double type, since the matlab net tructure holds its info with
            //double precision.      
            MxArrayHandler<double> iw, ib;
            mxArray *lw;
            mxArray *lb;
            const REAL ***savedW = net->getSavedWeights();
            const REAL **savedB = net->getSavedBias();
    
            //Getting the bias cells vector.
            lb = mxGetField(outNet, 0, "b");
    
            //Processing first the input layer.
            iw = mxGetCell(mxGetField(outNet, 0, "IW"), 0);
            ib = mxGetCell(lb, 0);
    
            DEBUG2("### Weights and Bias of the Best Train #######");
            for (unsigned i=0; i<numNodes[1]; i++)
            {
                ib(i) = static_cast<double>(savedB[0][i]);
                DEBUG2("b[0][" << i << "] = " << static_cast<double>(savedB[0][i]));
                for (unsigned j=0; j<numNodes[0]; j++)
                {
                    iw(i,j) = static_cast<double>(savedW[0][i][j]);
                    DEBUG2("w[" << 0 << "][" << i << "][" << j << "] = " << static_cast<double>(savedW[0][i][j]));
                }
            }
    
            //Processing the other layers.
            //Getting the weights cell matrix.
            lw = mxGetField(outNet, 0, "LW");
    
            for (unsigned i=1; i<(numNodes.size()-1); i++)
            {
                iw = mxGetCell(lw, iw.getPos(i,(i-1), mxGetM(lw)));
                ib = mxGetCell(lb, i);
          
                for (unsigned j=0; j<numNodes[(i+1)]; j++)
                {
                    ib(j) = static_cast<double>(savedB[i][j]);
                    DEBUG2("b[" << i << "][" << j << "] = " << static_cast<double>(savedB[i][j]));
                    for (unsigned k=0; k<numNodes[i]; k++)
                    {
                        iw(j,k) = static_cast<double>(savedW[i][j][k]);
                        DEBUG2("w[" << i << "][" << j << "][" << k << "] = " << static_cast<double>(savedW[i][j][k]));
                    }
                }
            }
            DEBUG2("### End of the Weights and Bias of the Best Train #######");
        }
};
#endif
