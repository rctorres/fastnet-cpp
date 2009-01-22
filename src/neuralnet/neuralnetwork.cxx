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
  NeuralNetwork::NeuralNetwork(const vector<unsigned> &nodesDist, const vector<string> &trfFunction)
  {
    nLayers = nodesDist.size();

    //Initializing the values, in case of error.
    weights = NULL;
    bias = NULL;
    layerOutputs = NULL;
    nNodes = NULL;
    trfFunc = NULL;
    activeNodes = NULL;
    frozenNode = NULL;
    usingBias = NULL;
  
    const unsigned size = nLayers - 1;

    try
    {
      //Allocating the nNodes vector.
      nNodes = new unsigned [nLayers];

      //Allocating the nodes range vector.
      activeNodes = new NodesRange [nLayers];

      //Initializing the nNodes an range vectors.
      for (unsigned i=0; i<nLayers; i++)
      {
        nNodes[i] = nodesDist[i];
        activeNodes[i].init = 0;
        activeNodes[i].end = nNodes[i];
      }

      //Reading and setting the transfer function in each layer.
      trfFunc = new TRF_FUNC_PTR [size];
      
      for (unsigned i=0; i<size; i++)
      {
        if (!strcmp(trfFunction[i].c_str(), TGH_ID))
        {
          trfFunc[i] = &NeuralNetwork::hyperbolicTangent;
        }
        else if (!strcmp(trfFunction[i].c_str(), LIN_ID))
        {
          trfFunc[i] = &NeuralNetwork::linear;
        }
        else
        {
          throw "Transfer function not specified!";
        }
      }

      //Allocating the bias matrix.
      bias = new REAL* [size];
      //Initiallizing with NULL in case of an future error on allocating memory.
      for (unsigned i=0; i<size; i++) bias[i] = NULL;
      //Allocating the matrix's collumns.
      for (unsigned i=0; i<size; i++) bias[i] = new REAL [nNodes[i+1]];

      //Allocating the weight matrix.
      weights = new REAL** [size];
      for (unsigned i=0; i<size; i++) weights[i] = NULL;
      for (unsigned i=0; i<size; i++)
      {
        weights[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++) weights[i][j] = NULL;
        for (unsigned j=0; j<nNodes[i+1]; j++) weights[i][j] = new REAL [nNodes[i]];
      }

      //Allocating space for the hidden outputs (just if we have at least one
      //hidden layer). The first position in hidOutput will not
      //be allocated, since it will point directly to the event's input.
      layerOutputs = new REAL* [nLayers];
      for (unsigned i=0; i<nLayers; i++) layerOutputs[i] = NULL;
      for (unsigned i=1; i<nLayers; i++) layerOutputs[i] = new REAL [nNodes[i]];

      //Allocating the frozenNode matrix and initializing with false.
      frozenNode = new bool* [size];
      //Initiallizing with NULL in case of an future error on allocating memory.
      for (unsigned i=0; i<size; i++) frozenNode[i] = NULL;
      //Allocating the matrix's collumns.
      for (unsigned i=0; i<size; i++)
      {
        frozenNode[i] = new bool [nNodes[i+1]];
        //Bellow, it's not (i+1) since the freezeNode matrix
        //has 'size' collums, and not 'nLayers' collums.
        setFrozen(i, false);
      }

      //Allocating the using bias vector and initializing with true.
      usingBias = new bool [size];
      setUsingBias(true);
    }
    catch (bad_alloc xa)
    {
      throw;
    }
  }

  NeuralNetwork::NeuralNetwork(const NeuralNetwork &net)
  {
    try
    {
      nLayers = net.nLayers;

      nNodes = new unsigned [nLayers];
      memcpy(nNodes, net.nNodes, nLayers*sizeof(unsigned));
      
      activeNodes = new NodesRange [nLayers];
      memcpy(activeNodes, net.activeNodes, nLayers*sizeof(NodesRange));
      
      layerOutputs = new REAL* [nLayers];
      layerOutputs[0] = net.layerOutputs[0]; // This will be a pointer to the input event.
    
      const unsigned size = nLayers-1;
      
      trfFunc = new TRF_FUNC_PTR [size];
      memcpy(trfFunc, net.trfFunc, size*sizeof(TRF_FUNC_PTR));
      
      usingBias = new bool [size];
      memcpy(usingBias, net.usingBias, size*sizeof(bool));
      
      bias = new REAL* [size];
      frozenNode = new bool* [size];
      weights = new REAL** [size];

      for (unsigned i=0; i<size; i++)
      {
        bias[i] = new REAL [nNodes[i+1]];
        memcpy(bias[i], net.bias[i], nNodes[i+1]*sizeof(REAL));
        
        frozenNode[i] = new bool [nNodes[i+1]];
        memcpy(frozenNode[i], net.frozenNode[i], nNodes[i+1]*sizeof(bool));
        
        layerOutputs[i+1] = new REAL [nNodes[i+1]];
        memcpy(layerOutputs[i+1], net.layerOutputs[i+1], nNodes[i+1]*sizeof(REAL));

        weights[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++)
        {
          weights[i][j] = new REAL [nNodes[i]];
          memcpy(weights[i][j], net.weights[i][j], nNodes[i]*sizeof(REAL));
        }
      }
    }
    catch (bad_alloc xa)
    {
      throw;
    }    
  }

  NeuralNetwork::NeuralNetwork(const mxArray *netStr)
  {
  }


  NeuralNetwork::~NeuralNetwork()
  {
    const unsigned size = nLayers - 1;

    // Deallocating the transfer function vector.
    if (trfFunc) delete [] trfFunc;

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

    // Deallocating the nNodes vector.
    if (nNodes) delete [] nNodes;

    // Deallocating the activeNodes vector.
    if (activeNodes) delete [] activeNodes;

    // Deallocating the frozenNode matrix.
    if (frozenNode)
    {
      for (unsigned i=0; i<size; i++)
      {
        if (frozenNode[i]) delete [] frozenNode[i];
      }

      delete [] frozenNode;
    }

    //Deallocating the usingBias vector.
    if (usingBias) delete [] usingBias;
  }
  
  
  void NeuralNetwork::showInfo(ostream &str) const
  {
    str << "NEURAL NETWORK CONFIGURATION INFO" << endl << endl;
    str << "Number of Layers (including the input): " << nLayers << endl << endl;
    
    for (unsigned i=0; i<nLayers; i++)
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
    const unsigned size = (nLayers - 1);

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

    for (unsigned i=0; i<(nLayers - 1); i++)
    {
      for (unsigned j=activeNodes[i+1].init; j<activeNodes[i+1].end; j++)
      {
        //Initializing the bias (if the layer is not using bias, the value is set to zero).
        bias[i][j] = (usingBias[i]) ? (2*initWeightRange*((static_cast<double>(rand() % 101)) / 100)) - initWeightRange : 0;

        //Initializing the weights.
        for (unsigned k=activeNodes[i].init; k<activeNodes[i].end; k++)
        {
          weights[i][j][k] = (2*initWeightRange*((static_cast<double>(rand() % 101)) / 100)) - initWeightRange;
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
    int size = (nLayers-1);
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
      for (int i=0; i<(nLayers-1); i++)
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
      for (int i=0; i<(nLayers-1); i++)
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


  void NeuralNetwork::setUsingBias(unsigned layer, bool val)
  {
    usingBias[layer] = val;
    
    //If not using layers, we assign the biases values
    //in the layer to 0.
    if(!usingBias[layer])
    {
      for (int i=activeNodes[(layer+1)].init; i<activeNodes[(layer+1)].end; i++)
      {
        bias[layer][i] = 0;
      }
    }
  }
}
