/** 
@file  neuralnetwork.cpp
@brief NeuralNetwork class implementation file.
*/

#include <iostream>
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
      memcpy(savedB[i], net.savedB[i], nNodes[i+1]*sizeof(REAL));
      memcpy(frozenNode[i], net.frozenNode[i], nNodes[i+1]*sizeof(bool));
      memcpy(layerOutputs[i+1], net.layerOutputs[i+1], nNodes[i+1]*sizeof(REAL));
      for (unsigned j=0; j<nNodes[i+1]; j++)
      {
        memcpy(weights[i][j], net.weights[i][j], nNodes[i]*sizeof(REAL));
        memcpy(savedW[i][j], net.savedW[i][j], nNodes[i]*sizeof(REAL));
      }
    }   
  }


  NeuralNetwork::NeuralNetwork(const mxArray *netStr)
  {
    DEBUG0("Initializing the NeuralNetwork class from a Matlab Network structure.");

    //Getting the number of nodes in the input layer.
    this->nNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(mxGetCell(mxGetField(netStr, 0, "inputs"), 0), 0, "size"))));
    DEBUG1("Number of nodes in layer 0: " << nNodes[0]);
    
    //Getting the number of nodes and transfer function in each layer:
    const mxArray *layers = mxGetField(netStr, 0, "layers");
    for (size_t i=0; i<mxGetM(layers); i++)
    {
      const mxArray *layer = mxGetCell(layers, i);
      this->nNodes.push_back(static_cast<unsigned>(mxGetScalar(mxGetField(layer, 0, "size"))));
      DEBUG1("Number of nodes in layer " << (i+1) << ": " << nNodes[(i+1)]);
      const string transFunction = mxArrayToString(mxGetField(layer, 0, "transferFcn"));
      if (transFunction == TGH_ID)
      {
        this->trfFunc.push_back(&NeuralNetwork::hyperbolicTangent);
        DEBUG1("Transfer function in layer " << (i+1) << ": tanh");
      }
      else if (transFunction == LIN_ID)
      {
        this->trfFunc.push_back(&NeuralNetwork::linear);
        DEBUG1("Transfer function in layer " << (i+1) << ": purelin");
      }
      else throw "Transfer function not specified!";
    }
    
    //Allocating the memory for the other values.
    try {allocateSpace();}
    catch (bad_alloc xa) {throw;}
    
     // This will be a pointer to the input event.
    layerOutputs[0] = NULL;
    
    //Taking the weights and values info.
    readWeights(netStr);
    
    //The savedW and savedB matrices are initialized with the read weights and biases values.
    saveBestTrain();

    //Getting the active nodes of the input layer.
    const mxArray *userData = mxGetField(mxGetCell(mxGetField(netStr, 0, "inputs"), 0), 0, "userdata");
    NodesRange aux;
    aux.init = static_cast<unsigned>(mxGetScalar(mxGetField(userData, 0, "initNode"))) - 1;
    aux.end = static_cast<unsigned>(mxGetScalar(mxGetField(userData, 0, "endNode")));
    if ( (aux.init <= aux.end) && (aux.end <= this->nNodes[0]) ) this->activeNodes.push_back(aux);
    else throw "Invalid nodes init or end values!";
    DEBUG1("Active nodes in layer 0 goes from " << this->activeNodes[0].init << " to " << this->activeNodes[0].end);
    
    //Verifying if there are frozen nodes and seting them, if so.
    // This loop also set if a given layer is not using bias and the start and end
    //nodes of each layer.
    for (unsigned i=0; i<mxGetM(layers); i++)
    {
      //Getting the nodes range information.
      const mxArray *userData = mxGetField(mxGetCell(layers, i), 0, "userdata");
      NodesRange aux;
      aux.init = static_cast<unsigned>(mxGetScalar(mxGetField(userData, 0, "initNode"))) - 1;
      aux.end = static_cast<unsigned>(mxGetScalar(mxGetField(userData, 0, "endNode")));
      if ( (aux.init <= aux.end) && (aux.end <= this->nNodes[(i+1)]) ) this->activeNodes.push_back(aux);
      else throw "Invalid nodes init or end values!";
      DEBUG1("Active nodes in layer " << (i+1) << " goes from " << this->activeNodes[(i+1)].init << " to " << this->activeNodes[(i+1)].end);
      
      //Getting the using bias information.
      this->usingBias.push_back(static_cast<bool>(mxGetScalar(mxGetField(userData, 0, "usingBias"))));
      DEBUG1("Layer " << (i+1) << " is using bias? " << this->usingBias[i]);
      
      // For the frozen nodes, we first initialize them all as unfrozen.
      setFrozen(i, false);
      
      //Getting from Matlab any possible frozen nodes.
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
      for (unsigned j=0; j<nNodes[0]; j++)
      {
        weights[0][i][j] = static_cast<REAL>(iw(i,j));
        DEBUG1("Weight[0][" << i << "][" << j << "] = " << weights[0][i][j]);
      }
      bias[0][i] = static_cast<REAL>(ib(i));
      DEBUG1("Bias[0][" << i << "] = " << bias[0][i]);
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
          DEBUG1("Weight[" << i << "][" << j << "][" << k << "] = " << weights[i][j][k]);
        }
        bias[i][j] = static_cast<REAL>(ib(j));
        DEBUG1("Bias[" << i << "][" << j << "] = " << bias[i][j]);
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
      savedB = new REAL* [size];
      frozenNode = new bool* [size];
      weights = new REAL** [size];
      savedW = new REAL** [size];

      for (unsigned i=0; i<size; i++)
      {
        bias[i] = new REAL [nNodes[i+1]];
        savedB[i] = new REAL [nNodes[i+1]];
        frozenNode[i] = new bool [nNodes[i+1]];
        layerOutputs[i+1] = new REAL [nNodes[i+1]];
        weights[i] = new REAL* [nNodes[i+1]];
        savedW[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++)
        {
          weights[i][j] = new REAL [nNodes[i]];
          savedW[i][j] = new REAL [nNodes[i]];
        }
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
    releaseMatrix(savedB);
    releaseMatrix(weights);
    releaseMatrix(savedW);
    
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

  void NeuralNetwork::flushBestTrainWeights(mxArray *outNet) const
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
    
    for (unsigned i=0; i<nNodes[1]; i++)
    {
      for (unsigned j=0; j<nNodes[0]; j++) iw(i,j) = static_cast<double>(savedW[0][i][j]);
      ib(i) = static_cast<double>(savedB[0][i]);
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
        for (unsigned k=0; k<nNodes[i]; k++) iw(j,k) = static_cast<double>(savedW[i][j][k]);
        ib(j) = static_cast<double>(savedB[i][j]);
      }
    }
  }
}
