/** 
@file  rprop.cpp
@brief The Resilient BackPropagation (RProp) class definition.
*/

#include <vector>
#include <string>

#include "fastnet/neuralnet/rprop.h"
#include "fastnet/reporter/Reporter.h"


using namespace std;

namespace FastNet
{
  RProp::RProp(const RProp &net) : Backpropagation(net)
  {
    deltaMax = net.deltaMax;
    deltaMin = net.deltaMin;
    incEta = net.incEta;
    decEta = net.decEta;
    initEta = net.initEta;

    try {allocateSpace();}
    catch (bad_alloc xa) {throw;}

    for (unsigned i=0; i<(nNodes.size() - 1); i++)
    {
      memcpy(prev_db[i], net.prev_db[i], nNodes[i+1]*sizeof(REAL));
      memcpy(delta_b[i], net.delta_b[i], nNodes[i+1]*sizeof(REAL));
      for (unsigned j=0; j<nNodes[i+1]; j++) 
      {
        memcpy(prev_dw[i][j], net.prev_dw[i][j], nNodes[i]*sizeof(REAL));
        memcpy(delta_w[i][j], net.delta_w[i][j], nNodes[i]*sizeof(REAL));
      }
    }
  }

  RProp::RProp(const mxArray *netStr) : Backpropagation(netStr)
  {
    DEBUG0("Initializing the RProp class from a Matlab Network structure.");
    const mxArray *trnParam =  mxGetField(netStr, 0, "trainParam");
    if (mxGetField(trnParam, 0, "deltamax")) this->deltaMax = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "deltamax")));
    else this->deltaMax = 50.0;
    if (mxGetField(trnParam, 0, "min_grad")) this->deltaMin = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "min_grad")));
    else this->deltaMin = 1E-6;
    if (mxGetField(trnParam, 0, "delt_inc")) this->incEta = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "delt_inc")));
    else this->incEta = 1.10;
    if (mxGetField(trnParam, 0, "delt_dec")) this->decEta = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "delt_dec")));
    else this->decEta = 0.5;
    if (mxGetField(trnParam, 0, "delta0")) this->initEta = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "delta0")));
    else this->initEta = 0.1;

    try {allocateSpace();}
    catch (bad_alloc xa) {throw;}

    //Initializing the dynamically allocated values.
    for (unsigned i=0; i<(nNodes.size() - 1); i++)
    {
      for (unsigned j=0; j<nNodes[i+1]; j++) 
      {
        prev_db[i][j] = 0.;
        delta_b[i][j] = this->initEta;
        
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          prev_dw[i][j][k] = 0.;
          delta_w[i][j][k] = this->initEta;
        }
      }
    }
  }


  void RProp::allocateSpace()
  {
    const unsigned size = nNodes.size() - 1;
    
    try
    {
      prev_db = new REAL* [size];
      delta_b = new REAL* [size];
      prev_dw = new REAL** [size];
      delta_w = new REAL** [size];
      for (unsigned i=0; i<size; i++)
      {
        prev_db[i] = new REAL [nNodes[i+1]];
        delta_b[i] = new REAL [nNodes[i+1]];
        prev_dw[i] = new REAL* [nNodes[i+1]];
        delta_w[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++) 
        {
          prev_dw[i][j] = new REAL [nNodes[i]];
          delta_w[i][j] = new REAL [nNodes[i]];
        }
      }
    }
    catch (bad_alloc xa)
    {
      throw;
    }    
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
    for (unsigned i=0; i<(nNodes.size()-1); i++)
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
    str << "Maximum allowed learning rate value (deltaMax) = " << deltaMax << endl;
    str << "Minimum allowed learning rate value (deltaMin) = " << deltaMin << endl;
    str << "Learning rate increasing factor (incEta) = " << incEta << endl;
    str << "Learning rate decreasing factor (decEta) = " << decEta << endl;
    str << "Initial learning rate value (initEta) = " << initEta << endl;
  }
}
