/** 
@file  rprop.cpp
@brief The Resilient BackPropagation (RProp) class definition.
*/

#include <vector>
#include <string>

#include "fastnet/neuralnet/rprop.h"

using namespace std;

namespace FastNet
{
  RProp::RProp(const vector<unsigned> &nodesDist, const vector<string> &trfFunction) : Backpropagation(nodesDist, trfFunction)
  {
    deltaMax = 50.0;
    deltaMin = 1E-6;
    incEta = 1.10;
    decEta = 0.5;
    initEta = 0.1;
    
    const unsigned size = nLayers - 1;
    
    try
    {
      //Allocating the biases matrix.
      prev_db = new REAL* [size];
      delta_b = new REAL* [size];
      
      //Initiallizing with NULL in case of an future error on allocating memory.
      for (unsigned i=0; i<size; i++) prev_db[i] = delta_b[i] = NULL;

      //Allocating the matrix's collumns.
      for (unsigned i=0; i<size; i++)
      {
        prev_db[i] = new REAL [nNodes[i+1]];
        delta_b[i] = new REAL [nNodes[i+1]];
        
        for (unsigned j=0; j<nNodes[i+1]; j++)
        {
          prev_db[i][j] = 0;
          delta_b[i][j] = initEta;
        }
      }

      //Allocating the weight matrices.
      prev_dw = new REAL** [size];
      delta_w = new REAL** [size];

      for (unsigned i=0; i<size; i++) prev_dw[i] = delta_w[i] = NULL;
      for (unsigned i=0; i<size; i++)
      {
        prev_dw[i] = new REAL* [nNodes[i+1]];
        delta_w[i] = new REAL* [nNodes[i+1]];

        for (unsigned j=0; j<nNodes[i+1]; j++) prev_dw[i][j] = delta_w[i][j] = NULL;
        for (unsigned j=0; j<nNodes[i+1]; j++) 
        {
          prev_dw[i][j] = new REAL [nNodes[i]];
          delta_w[i][j] = new REAL [nNodes[i]];

          for (unsigned k=0; k<nNodes[i]; k++)
          {
            prev_dw[i][j][k] = 0;
            delta_w[i][j][k] = initEta;
          }
        }
      }
    }
    catch (bad_alloc xa)
    {
      throw;
    }
  }


  RProp::RProp(const RProp &net) : Backpropagation(net)
  {
    deltaMax = net.deltaMax;
    deltaMin = net.deltaMin;
    incEta = net.incEta;
    decEta = net.decEta;
    initEta = net.initEta;
    
    const unsigned size = nLayers - 1;
    
    try
    {
      prev_db = new REAL* [size];
      delta_b = new REAL* [size];
      prev_dw = new REAL** [size];
      delta_w = new REAL** [size];
      for (unsigned i=0; i<size; i++)
      {
        prev_db[i] = new REAL [nNodes[i+1]];
        memcpy(prev_db[i], net.prev_db[i], nNodes[i+1]*sizeof(REAL));

        delta_b[i] = new REAL [nNodes[i+1]];
        memcpy(delta_b[i], net.delta_b[i], nNodes[i+1]*sizeof(REAL));

        prev_dw[i] = new REAL* [nNodes[i+1]];
        delta_w[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++) 
        {
          prev_dw[i][j] = new REAL [nNodes[i]];
          memcpy(prev_dw[i][j], net.prev_dw[i][j], nNodes[i]*sizeof(REAL));

          delta_w[i][j] = new REAL [nNodes[i]];
          memcpy(delta_w[i][j], net.delta_w[i][j], nNodes[i]*sizeof(REAL));
        }
      }
    }
    catch (bad_alloc xa)
    {
      throw;
    }    
  }

  RProp::RProp(const mxArray *netStr) : Backpropagation(netStr)
  {
  }

  
  RProp::~RProp()
  {
    // Deallocating the delta bias matrix.
    releaseMatrix(prev_db);
    releaseMatrix(delta_b);

    // Deallocating the delta weights matrix.
    releaseMatrix(prev_dw);
    releaseMatrix(delta_w);
  }



  void RProp::updateWeights()
  {
    for (unsigned i=0; i<(nLayers-1); i++)
    {
      for (unsigned j=activeNodes[(i+1)].init; j<activeNodes[(i+1)].end; j++)
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
            updateW(delta_w[i][j][k], dw[i][j][k], prev_dw[i][j][k], weights[i][j][k]);
          }
        
          if (usingBias[i]) updateW(delta_b[i][j], db[i][j], prev_db[i][j], bias[i][j]);
          else bias[i][j] = 0;
        }
      }
    }
  }


  inline void RProp::updateW(REAL &delta, REAL &d, REAL &prev_d, REAL &w)
  {
    const REAL val = prev_d * d;
          
    if (val > 0.)
    {
      delta = min((delta*incEta), deltaMax);
    }
    else if (val < 0.)
    {
      delta = max((delta*decEta), deltaMin);
    }

    w += (sign(d) * delta);
    prev_d = d;
    d = 0;
  }
  
  
  void RProp::showInfo(ostream &str) const
  {
    NeuralNetwork::showInfo(str);
    str << "TRAINING ALGORITHM INFORMATION" << endl;
    str << "Training algorithm: Resilient Backpropagation" << endl;
  }
}
