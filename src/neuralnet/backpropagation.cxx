/**
@file  backpropagation.cpp
@brief The BackPropagation class definition.
*/

#include <vector>
#include <string>
#include <cstdlib>
#include <typeinfo>
#include <sstream>

#include "fastnet/neuralnet/backpropagation.h"
#include "fastnet/sys/Reporter.h"
#include "fastnet/sys/mxhandler.hxx"

namespace FastNet
{
  Backpropagation::Backpropagation(const Backpropagation &net) : NeuralNetwork(net)
  { 
    try {allocateSpace(net.nNodes);}
    catch (bad_alloc xa) {throw;}
    (*this) = net; 
  }


  void Backpropagation::operator=(const Backpropagation &net)
  { 
    DEBUG1("Attributing all values using assignment operator for Backpropagation class");
    NeuralNetwork::operator=(net);
    
    learningRate = net.learningRate;
    decFactor = net.decFactor;

    for (unsigned i=0; i<(nNodes.size() - 1); i++)
    {
      memcpy(savedB[i], net.savedB[i], nNodes[i+1]*sizeof(REAL));
      memcpy(frozenNode[i], net.frozenNode[i], nNodes[i+1]*sizeof(bool));
      memcpy(db[i], net.db[i], nNodes[i+1]*sizeof(REAL));
      memcpy(sigma[i], net.sigma[i], nNodes[i+1]*sizeof(REAL));
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        memcpy(dw[i][j], net.dw[i][j], nNodes[i]*sizeof(REAL));
        memcpy(savedW[i][j], net.savedW[i][j], nNodes[i]*sizeof(REAL));
      }
    }
  }
  

  Backpropagation::Backpropagation(const mxArray *netStr) : NeuralNetwork(netStr)
  {
    DEBUG1("Initializing the Backpropagation class from a Matlab Network structure.");

    //We first test whether the values exists, otherwise, we use default ones.
    const mxArray *trnParam =  mxGetField(netStr, 0, "trainParam");
    if (mxGetField(trnParam, 0, "lr")) this->learningRate = static_cast<REAL>(abs(mxGetScalar(mxGetField(trnParam, 0, "lr"))));
    else this->learningRate = 0.05;
    if (mxGetField(trnParam, 0, "decFactor")) this->decFactor = static_cast<REAL>(abs(mxGetScalar(mxGetField(trnParam, 0, "decFactor"))));
    else this->decFactor = 1;

    try {allocateSpace(nNodes);}
    catch (bad_alloc xa) {throw;}

    //The savedW and savedB matrices are initialized with the read weights and biases values.
    saveBestTrain();

    //Verifying if there are frozen nodes and seting them.
    const mxArray *layers = mxGetField(netStr, 0, "layers");
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      // For the frozen nodes, we first initialize them all as unfrozen.
      setFrozen(i, false);
      
      //Getting from Matlab any possible frozen nodes.
      const mxArray *userData = mxGetField(mxGetCell(layers, i), 0, "userdata");
      const mxArray *matFNodes = mxGetField(userData, 0, "frozenNodes");
      const double *fNodes = mxGetPr(matFNodes);
      for (unsigned j=0; j<mxGetN(matFNodes); j++)
      {
        const unsigned node = static_cast<unsigned>(fNodes[j]) - 1;
        if (node < nNodes[(i+1)]) setFrozen(i, node, true);
        else throw "Node to be frozen is invalid!";
      }

      //Initializing dw and db.
      for (unsigned j=0; j<nNodes[i+1]; j++) 
      {
        this->db[i][j] = 0.;
        this->sigma[i][j] = 0.;
        for (unsigned k=0; k<nNodes[i]; k++) this->dw[i][j][k] = 0.;
      }
    }    
  }


  void Backpropagation::allocateSpace(const vector<unsigned> &nNodes)
  {
    const unsigned size = nNodes.size() - 1;
    try
    {
      frozenNode = new bool* [size];
      savedB = new REAL* [size];
      savedW = new REAL** [size];
      db = new REAL* [size];
      sigma = new REAL* [size];
      dw = new REAL** [size];
      for (unsigned i=0; i<size; i++)
      {
        savedW[i] = new REAL* [nNodes[i+1]];
        savedB[i] = new REAL [nNodes[i+1]];
        frozenNode[i] = new bool [nNodes[i+1]];
        db[i] = new REAL [nNodes[i+1]];
        sigma[i] = new REAL [nNodes[i+1]];
        dw[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++)
        {
          dw[i][j] = new REAL [nNodes[i]];
          savedW[i][j] = new REAL [nNodes[i]];
        }
      }
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
    releaseMatrix(savedB);
    releaseMatrix(savedW);

    // Deallocating the frozenNode matrix.
    if (frozenNode)
    {
      for (unsigned i=0; i<(nNodes.size()-1); i++) if (frozenNode[i]) delete [] frozenNode[i];
      delete [] frozenNode;
    }

  }

  void Backpropagation::retropropagateError(const REAL *output, const REAL *target)
  {
    const unsigned size = nNodes.size() - 1;

    for (unsigned i=0; i<nNodes[size]; i++) sigma[size-1][i] = (target[i] - output[i]) * CALL_TRF_FUNC(trfFunc[size-1])(output[i], true);

    //Retropropagating the error.
    for (int i=(size-2); i>=0; i--)
    {
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        sigma[i][j] = 0;

        for (unsigned k=0; k<nNodes[i+2]; k++)
        {
          sigma[i][j] += sigma[i+1][k] * weights[(i+1)][k][j];
        }

        sigma[i][j] *= CALL_TRF_FUNC(trfFunc[i])(layerOutputs[i+1][j], true);
      }
    }
  }
  

  void Backpropagation::calculateNewWeights(const REAL *output, const REAL *target)
  {
    const unsigned size = nNodes.size() - 1;

    retropropagateError(output, target);

    //Accumulating the deltas.
    for (unsigned i=0; i<size; i++)
    {
      for (unsigned j=0; j<nNodes[(i+1)]; j++)
      {
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          dw[i][j][k] += (sigma[i][j] * layerOutputs[i][k]);
        }

        db[i][j] += (sigma[i][j]);
      }
    }
  }


  void Backpropagation::addToGradient(const Backpropagation &net)
  {
    //Accumulating the deltas.
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      for (unsigned j=0; j<nNodes[(i+1)]; j++)
      {
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          dw[i][j][k] += net.dw[i][j][k];
        }
        db[i][j] += net.db[i][j];
      }
    }
  }

  void Backpropagation::updateWeights(const unsigned numEvents)
  {
    const REAL val = 1. / static_cast<REAL>(numEvents);
    
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      for (unsigned j=0; j<nNodes[(i+1)]; j++)
      {
        //If the node is frozen, we just reset the accumulators,
        //otherwise, we actually train the weights connected to it.
        if (frozenNode[i][j])
        {
          DEBUG2("Skipping updating node " << j << " from hidden layer " << i << ", since it is frozen!");
          for (unsigned k=0; k<nNodes[i]; k++) dw[i][j][k] = 0;
          if (usingBias[i]) db[i][j] = 0;
          else bias[i][j] = 0;
        }
        else
        {
          for (unsigned k=0; k<nNodes[i]; k++)
          {
            weights[i][j][k] += (learningRate * val * dw[i][j][k]);
            dw[i][j][k] = 0;
          }

          if (usingBias[i])
          {
            bias[i][j] += (learningRate * val * db[i][j]);
            db[i][j] = 0;
          }
          else
          {
            bias[i][j] = 0;
          }
        }
      }
    }
  }


  void Backpropagation::showInfo() const
  {
    NeuralNetwork::showInfo();
    REPORT("TRAINING ALGORITHM INFORMATION:");
    REPORT("Training algorithm : Gradient Descent");
    REPORT("Learning rate      : " << learningRate);
    REPORT("Decreasing factor  : " << decFactor);
        
    for (unsigned i=0; i<nNodes.size()-1; i++) 
    {
      std::ostringstream aux;
      aux << "Frozen Nodes in hidden layer " << i << ":";
      bool frozen = false;
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        if (frozenNode[i][j])
        {
          aux << " " << j;
          frozen = true;
        }
      }
      if (!frozen) aux << " NONE";
      REPORT(aux.str());
    }
  }

  void Backpropagation::initWeights(REAL initWeightRange)
  {
    // Choosing a seed based on system time.
    srand(time(NULL) % 10000000);

    for (unsigned i=0; i<(nNodes.size() - 1); i++)
    {
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        //Initializing the bias (if the layer is not using bias, the value is set to zero).
        bias[i][j] = (usingBias[i]) ? (2*initWeightRange*((static_cast<REAL>(rand() % 101)) / 100)) - initWeightRange : 0;

        //Initializing the weights.
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          weights[i][j][k] = (2*initWeightRange*((static_cast<REAL>(rand() % 101)) / 100)) - initWeightRange;
        }
      }
    }
  }


  bool Backpropagation::isFrozen(unsigned layer) const
  {
    for (int i=0; i<nNodes[layer+1]; i++)
    {
      if (!frozenNode[layer][i]) return false;
    }

    return true;
  }


  inline REAL Backpropagation::applySupervisedInput(const REAL *input, const REAL *target, const REAL* &output)
  {
    int size = (nNodes.size()-1);
    REAL error = 0;

    //Propagating the input.
    output = propagateInput(input);
      
    //Calculating the error.
    for (int i=0; i<nNodes[size]; i++) error += SQR(target[i] - output[i]);

    //Returning the MSE
    return (error / nNodes[size]);
  }

  void Backpropagation::flushBestTrainWeights(mxArray *outNet) const
  {
    // It must be of double type, since the matlab net tructure holds its info with
    //double precision.      
    MxArrayHandler<double> iw, ib;
    mxArray *lw;
    mxArray *lb;
    
    //Getting the bias cells vector.
    lb = mxGetField(outNet, 0, "b");
    
    //Processing first the input layer.
    iw = mxGetCell(mxGetField(outNet, 0, "IW"), 0);
    ib = mxGetCell(lb, 0);
    
    DEBUG2("### Weights and Bias of the Best Train #######");
    for (unsigned i=0; i<nNodes[1]; i++)
    {
      ib(i) = static_cast<double>(savedB[0][i]);
      DEBUG2("b[0][" << i << "] = " << static_cast<double>(savedB[0][i]));
      for (unsigned j=0; j<nNodes[0]; j++)
      {
        iw(i,j) = static_cast<double>(savedW[0][i][j]);
        DEBUG2("w[" << 0 << "][" << i << "][" << j << "] = " << static_cast<double>(savedW[0][i][j]));
      }
    }
    
    //Processing the other layers.
    //Getting the weights cell matrix.
    lw = mxGetField(outNet, 0, "LW");
    
    for (unsigned i=1; i<(nNodes.size()-1); i++)
    {
      iw = mxGetCell(lw, iw.getPos(i,(i-1), mxGetM(lw)));
      ib = mxGetCell(lb, i);
          
      for (unsigned j=0; j<nNodes[(i+1)]; j++)
      {
        ib(j) = static_cast<double>(savedB[i][j]);
        DEBUG2("b[" << i << "][" << j << "] = " << static_cast<double>(savedB[i][j]));
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          iw(j,k) = static_cast<double>(savedW[i][j][k]);
          DEBUG2("w[" << i << "][" << j << "][" << k << "] = " << static_cast<double>(savedW[i][j][k]));
        }
      }
    }
    
    DEBUG2("### End of the Weights and Bias of the Best Train #######");
  }
}
