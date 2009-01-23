/** 
@file  neuralnetwork.cpp
@brief NeuralNetwork class implementation file.
*/

#include <ctime>
#include <new>
#include <cstdlib>
#include <vector>
#include <string>

#include "fastnet/neuralnet/neuralnetwork.h"

using namespace std;

namespace FastNet
{
  NeuralNetwork::NeuralNetwork(const NeuralNetwork &net)
  {
    nNodes.assign(net.nNodes.begin(), net.nNodes.end());
    activeNodes.assign(net.activeNodes.begin(), net.activeNodes.end());
    usingBias.assign(net.usingBias.begin(), net.usingBias.end());
    trfFunc.assign(net.trfFunc.begin(), net.trfFunc.end());
      
    //Allocating the memory for the other values.
    try {allocateSpace();}
    catch (bad_alloc xa) {throw;}

    layerOutputs[0] = net.layerOutputs[0]; // This will be a pointer to the input event.
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      memcpy(bias[i], net.bias[i], nNodes[i+1]*sizeof(REAL));
      memcpy(frozenNode[i], net.frozenNode[i], nNodes[i+1]*sizeof(bool));
      memcpy(layerOutputs[i+1], net.layerOutputs[i+1], nNodes[i+1]*sizeof(REAL));
      for (unsigned j=0; j<nNodes[i+1]; j++) memcpy(weights[i][j], net.weights[i][j], nNodes[i]*sizeof(REAL));
    }   
  }


  NeuralNetwork::NeuralNetwork(const mxArray *netStr)
  {
    //Getting the number of nodes in the input layer.
    this->nNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(mxGetCell(mxGetField(netStr, 0, "inputs"), 0), 0, "size"))));
    
    //Getting the number of nodes and transfer function in each layer:
    const mxArray *layers = mxGetField(netStr, 0, "layers");
    for (size_t i=0; i<mxGetM(layers); i++)
    {
      mxArray *layer = mxGetCell(layers, i);
      this->nNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(layer, 0, "size"))));
      const string transFunction = mxArrayToString(mxGetField(layer, 0, "transferFcn"));
      if (transFunction == TGH_ID) this->trfFunc.push_back(&NeuralNetwork::hyperbolicTangent);
      else if (transFunction == LIN_ID) this->trfFunc.push_back(&NeuralNetwork::linear);
      else throw "Transfer function not specified!";
    }
    
    //Allocating the memory for the other values.
    try {allocateSpace();}
    catch (bad_alloc xa) {throw;}
    
     // This will be a pointer to the input event.
    layerOutputs[0] = NULL;
    
    //Taking the weights and values info.
    readWeights(netStr);

    //Getting the active nodes of the input layer.
    const mxArray *userData = mxGetField(mxGetCell(mxGetField(netStr, 0, "inputs"), 0), 0, "userdata");
    NodesRange aux;
    aux.init = static_cast<unsigned>(mxGetScalar(mxGetField(userData, 0, "initNode"))) - 1;
    aux.end = static_cast<unsigned>(mxGetScalar(mxGetField(userData, 0, "endNode"))) - 1;
    if ( (aux.init <= aux.end) && (aux.end < nNodes[0]) ) this->activeNodes.push_back(aux);
    else throw "Invalid nodes init or end values!";
    
    //Verifying if there are frozen nodes and seting them, if so.
    // This loop also set if a given layer is not using bias and the start and end
    //nodes of each layer.
    for (unsigned i=0; i<mxGetM(layers); i++)
    {
      //Getting the nodes range information.
      const mxArray *userData = mxGetField(mxGetCell(layers, i), 0, "userdata");
      NodesRange aux;
      aux.init = static_cast<unsigned>(mxGetScalar(mxGetField(userData, 0, "initNode"))) - 1;
      aux.end = static_cast<unsigned>(mxGetScalar(mxGetField(userData, 0, "endNode"))) - 1;
      if ( (aux.init <= aux.end) && (aux.end < nNodes[(i+1)]) ) this->activeNodes.push_back(aux);
      else throw "Invalid nodes init or end values!";
      
      //Getting the using bias information.
      this->usingBias.push_back(static_cast<bool>(mxGetScalar(mxGetField(userData, 0, "usingBias"))));
      
      //Getting the frozen nodes information.
      const mxArray *matFNodes = mxGetField(userData, 0, "frozenNodes");
      const double *fNodes = mxGetPr(matFNodes);
      for (unsigned j=0; j<mxGetN(matFNodes); j++)
      {
        const unsigned node = static_cast<unsigned>(fNodes[j]) - 1;
        if (node < nNodes[(i+1)]) setFrozen(i, node, true);
        else throw "Node to be frozen is invalid!";
      }
    }
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
      for (unsigned j=0; j<nNodes[0]; j++) weights[0][i][j] = static_cast<REAL>(iw(i,j));
      bias[0][i] = static_cast<REAL>(ib(i));
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
        for (unsigned k=0; k<nNodes[i]; k++) weights[i][j][k] = static_cast<REAL>(iw(j,k));
        bias[i][j] = static_cast<REAL>(ib(j));
      }
    }
  }


  void NeuralNetwork::allocateSpace()
  {
    try
    {
      layerOutputs = new REAL* [nNodes.size()];
      layerOutputs[0] = NULL; // This will be a pointer to the input event.
    
      const unsigned size = nNodes.size()-1;
      
      bias = new REAL* [size];
      frozenNode = new bool* [size];
      weights = new REAL** [size];

      for (unsigned i=0; i<size; i++)
      {
        bias[i] = new REAL [nNodes[i+1]];
        frozenNode[i] = new bool [nNodes[i+1]];
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

    // Deallocating the frozenNode matrix.
    if (frozenNode)
    {
      for (unsigned i=0; i<size; i++)
      {
        if (frozenNode[i]) delete [] frozenNode[i];
      }

      delete [] frozenNode;
    }
  }
  
  
  void NeuralNetwork::showInfo(ostream &str) const
  {
    str << "NEURAL NETWORK CONFIGURATION INFO" << endl << endl;
    str << "Number of Layers (including the input): " << nNodes.size() << endl << endl;
    
    for (unsigned i=0; i<nNodes.size(); i++)
    {
      str << "Layer " << i << " Configuration:" << endl;
      str << "Number of Nodes   : " << nNodes[i] << endl;
      str << "Active Nodes      : from " << activeNodes[i].init << " to " << (activeNodes[i].end-1) << endl;
      
      if (i)
      {
        str << "Transfer function : ";
        if (trfFunc[(i-1)] == (&NeuralNetwork::hyperbolicTangent)) str << "tanh" << endl;
        else if (trfFunc[(i-1)] == (&NeuralNetwork::linear)) str << "purelin" << endl;
        else str << "UNKNOWN!" << endl;
        
        str << "Using bias        : ";
        if (usingBias[(i-1)]) str << "true" << endl;
        else  str << "false" << endl;
        
        str << "Frozen Nodes      :";
        bool frozen = false;
        for (int j=0; j<nNodes[i]; j++)
        {
          if (frozenNode[(i-1)][j])
          {
            str << " " << j;
            frozen = true;
          }
        }
        if (frozen) str << endl;
        else  str << " NONE" << endl;
      }
      
      str << endl;
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
      for (unsigned j=activeNodes[i+1].init; j<activeNodes[i+1].end; j++)
      {
        layerOutputs[i+1][j] = bias[i][j];
        
        for (unsigned k=activeNodes[i].init; k<activeNodes[i].end; k++)
        {
          layerOutputs[i+1][j] += layerOutputs[i][k] * weights[i][j][k];
        }

        layerOutputs[i+1][j] = CALL_TRF_FUNC(trfFunc[i])(layerOutputs[i+1][j], false);
      }
    }
    
    //Returning the network's output.
    return layerOutputs[size];
  }


  void NeuralNetwork::initWeights(REAL initWeightRange)
  {
    // Choosing a seed based on system time.
    srand(time(NULL) % 10000000);

    for (unsigned i=0; i<(nNodes.size() - 1); i++)
    {
      for (unsigned j=activeNodes[i+1].init; j<activeNodes[i+1].end; j++)
      {
        //Initializing the bias (if the layer is not using bias, the value is set to zero).
        bias[i][j] = (usingBias[i]) ? (2*initWeightRange*((static_cast<REAL>(rand() % 101)) / 100)) - initWeightRange : 0;

        //Initializing the weights.
        for (unsigned k=activeNodes[i].init; k<activeNodes[i].end; k++)
        {
          weights[i][j][k] = (2*initWeightRange*((static_cast<REAL>(rand() % 101)) / 100)) - initWeightRange;
        }
      }
    }
  }


  bool NeuralNetwork::isFrozen(unsigned layer) const
  {
    for (int i=activeNodes[layer+1].init; i<activeNodes[layer+1].end; i++)
    {
      if (!frozenNode[layer][i]) return false;
    }

    return true;
  }


  inline REAL NeuralNetwork::applySupervisedInput(const REAL *input, const REAL *target, const REAL* &output)
  {
    int size = (nNodes.size()-1);
    REAL error = 0;

    //Propagating the input.
    output = propagateInput(input);
      
    //Calculating the error.
    for (int i=activeNodes[size].init; i<activeNodes[size].end; i++) error += SQR(target[i] - output[i]);

    //Returning the MSE
    return (error / nNodes[size]);
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
      for (unsigned i=activeNodes[(layer+1)].init; i<activeNodes[(layer+1)].end; i++)
      {
        bias[layer][i] = 0;
      }
    }
  }
}
