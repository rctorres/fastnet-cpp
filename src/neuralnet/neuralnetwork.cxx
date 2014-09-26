/**
@file  neuralnetwork.cpp
@brief NeuralNetwork class implementation file.
*/

#include <iostream>
#include <new>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>

#include "fastnet/neuralnet/neuralnetwork.h"
#include "fastnet/sys/mxhandler.hxx"

using namespace std;

namespace FastNet
{
  NeuralNetwork::NeuralNetwork(const NeuralNetwork &net)
  {
    //Allocating the memory for the values.
    try {allocateSpace(net.nNodes);}
    catch (bad_alloc xa) {throw;}
    (*this) = net;
  }


  void NeuralNetwork::operator=(const NeuralNetwork &net)
  {
    nNodes.clear();
    usingBias.clear();
    trfFunc.clear();
    nNodes.assign(net.nNodes.begin(), net.nNodes.end());
    usingBias.assign(net.usingBias.begin(), net.usingBias.end());
    trfFunc.assign(net.trfFunc.begin(), net.trfFunc.end());
      
    layerOutputs[0] = net.layerOutputs[0]; // This will be a pointer to the input event.
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      memcpy(bias[i], net.bias[i], nNodes[i+1]*sizeof(REAL));
      memcpy(layerOutputs[i+1], net.layerOutputs[i+1], nNodes[i+1]*sizeof(REAL));
      for (unsigned j=0; j<nNodes[i+1]; j++) memcpy(weights[i][j], net.weights[i][j], nNodes[i]*sizeof(REAL));
    } 
  }


  NeuralNetwork::NeuralNetwork(const mxArray *netStr)
  {
    DEBUG1("Initializing the NeuralNetwork class from a Matlab Network structure.");

    //Getting the number of nodes in the input layer.
    this->nNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetProperty(mxGetCell(mxGetField(netStr, 0, "inputs"), 0), 0, "size"))));
    DEBUG2("Number of nodes in layer 0: " << nNodes[0]);
    
    //Getting the number of nodes and transfer function in each layer:
    const mxArray *layers = mxGetField(netStr, 0, "layers");
    for (size_t i=0; i<mxGetM(layers); i++)
    {
      const mxArray *layer = mxGetCell(layers, i);
      this->nNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetProperty(layer, 0, "size"))));
      DEBUG2("Number of nodes in layer " << (i+1) << ": " << nNodes[(i+1)]);
      const string transFunction = mxArrayToString(mxGetProperty(layer, 0, "transferFcn"));
      if (transFunction == TGH_ID)
      {
        this->trfFunc.push_back(&NeuralNetwork::hyperbolicTangent);
        DEBUG2("Transfer function in layer " << (i+1) << ": tanh");
      }
      else if (transFunction == LIN_ID)
      {
        this->trfFunc.push_back(&NeuralNetwork::linear);
        DEBUG2("Transfer function in layer " << (i+1) << ": purelin");
      }
      else throw "Transfer function not specified!";
    }
    
    //Allocating the memory for the other values.
    try {allocateSpace(nNodes);}
    catch (bad_alloc xa) {throw;}
    
     // This will be a pointer to the input event.
    layerOutputs[0] = NULL;
    
    //Getting the using bias information.
    for (unsigned i=0; i<mxGetM(layers); i++)
    {
      const mxArray *userData = mxGetProperty(mxGetCell(layers, i), 0, "userdata");
      this->usingBias.push_back(static_cast<bool>(mxGetScalar(mxGetField(userData, 0, "usingBias"))));
      DEBUG2("Layer " << (i+1) << " is using bias? " << this->usingBias[i]);
    }

    //Taking the weights and values info. This line must come after knowing wich layers won't
    //have biases.
    readWeights(netStr);
  }


  void NeuralNetwork::readWeights(const mxArray *mNet)
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


  void NeuralNetwork::allocateSpace(const vector<unsigned> &nNodes)
  {
    DEBUG2("Allocating all the space that the NeuralNetwork class will need.");
    try
    {
      layerOutputs = new REAL* [nNodes.size()];
      layerOutputs[0] = NULL; // This will be a pointer to the input event.
    
      const unsigned size = nNodes.size()-1;
      
      bias = new REAL* [size];
      weights = new REAL** [size];

      for (unsigned i=0; i<size; i++)
      {
        bias[i] = new REAL [nNodes[i+1]];
        layerOutputs[i+1] = new REAL [nNodes[i+1]];
        weights[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++) weights[i][j] = new REAL [nNodes[i]];
      }
    }
    catch (bad_alloc xa)
    {
      throw;
    }    
  }
  

  NeuralNetwork::~NeuralNetwork()
  {
    DEBUG2("Releasing all memory allocated by NeuralNetwork.");
    const unsigned size = nNodes.size() - 1;

    // Deallocating the bias and weight matrices.
    releaseMatrix(bias);
    releaseMatrix(weights);
    
    // Deallocating the hidden outputs matrix.
    if (layerOutputs)
    {
      for (unsigned i=1; i<size; i++)
      {
        if (layerOutputs[i]) delete [] layerOutputs[i];
      }

      delete [] layerOutputs;
    }
  }
  
  
  void NeuralNetwork::showInfo() const
  {
    REPORT("NEURAL NETWORK CONFIGURATION INFO");
    REPORT("Number of Layers (including the input): " << nNodes.size());
    
    for (unsigned i=0; i<nNodes.size(); i++)
    {
      REPORT("\nLayer " << i << " Configuration:");
      REPORT("Number of Nodes   : " << nNodes[i]);
      
      if (i)
      {
        std::ostringstream aux;
        aux << "Transfer function : ";
        if (trfFunc[(i-1)] == (&NeuralNetwork::hyperbolicTangent)) aux << "tanh";
        else if (trfFunc[(i-1)] == (&NeuralNetwork::linear)) aux << "purelin";
        else aux << "UNKNOWN!";

        aux << "\nUsing bias        : ";
        if (usingBias[(i-1)]) aux << "true";
        else  aux << "false";
        REPORT(aux.str());
      }      
    }
  }


  inline const REAL* NeuralNetwork::propagateInput(const REAL *input)
  {
    const unsigned size = (nNodes.size() - 1);

    //Placing the input. though we are removing the const' ness no changes are perfomed.
    layerOutputs[0] = const_cast<REAL*>(input);

    //Propagating the input through the network.
    for (unsigned i=0; i<size; i++)
    {
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        layerOutputs[i+1][j] = bias[i][j];
        
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          layerOutputs[i+1][j] += layerOutputs[i][k] * weights[i][j][k];
        }

        layerOutputs[i+1][j] = CALL_TRF_FUNC(trfFunc[i])(layerOutputs[i+1][j], false);
      }
    }
    
    //Returning the network's output.
    return layerOutputs[size];
  }
  

  void NeuralNetwork::releaseMatrix(REAL **b)
  {
    if (b)
    {
      for (int i=0; i<(nNodes.size()-1); i++)
      {
        if (b[i]) delete [] b[i];
      }
      delete [] b;
      b = NULL;
    }
  }


  void NeuralNetwork::releaseMatrix(REAL ***w)
  {
    if (w)
    {
      for (int i=0; i<(nNodes.size()-1); i++)
      {
        if (w[i])
        {
          for (int j=0; j<nNodes[i+1]; j++)
          {
            if (w[i][j]) delete [] w[i][j];
          }
          delete [] w[i];
        }
      }
      delete [] w;
      w = NULL;
    }
  }


  void NeuralNetwork::setUsingBias(const unsigned layer, const bool val)
  {
    usingBias[layer] = val;
    
    //If not using layers, we assign the biases values
    //in the layer to 0.
    if(!usingBias[layer])
    {
      for (unsigned i=0; i<nNodes[(layer+1)]; i++)
      {
        bias[layer][i] = 0;
      }
    }
  }
}
