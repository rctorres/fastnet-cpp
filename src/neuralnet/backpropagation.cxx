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
  

  Backpropagation::Backpropagation(const std::vector<unsigned> &nNodes, const std::vector<string> &trfFunc, 
                                                      const std::vector<bool> &usingBias,  const REAL learningRate ,
                                                      const REAL decFactor)  : NeuralNetwork(nNodes, trfFunc, usingBias)
    {
        DEBUG1("Initializing the Backpropagation class from scratch.");

        this->learningRate = learningRate;
        this->decFactor = decFactor;

        try {allocateSpace(nNodes);}
        catch (bad_alloc xa) {throw;}

        //Verifying if there are frozen nodes and seting them.
        for (unsigned i=0; i<(nNodes.size()-1); i++)
        {
            // For the frozen nodes, we first initialize them all as unfrozen.
            setFrozen(i, false);
      
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
    DEBUG2("Allocating all the space that the Backpropagation class will need.");
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
    DEBUG2("Releasing all memory allocated by Backpropagation.");
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

}
