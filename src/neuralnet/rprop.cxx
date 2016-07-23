/**  
@file  rprop.cpp
@brief The Resilient BackPropagation (RProp) class definition.
*/

#include <vector>
#include <string>
#include <typeinfo>

#include "fastnet/neuralnet/rprop.h"
#include "fastnet/sys/Reporter.h"


using namespace std;

namespace FastNet
{
  RProp::RProp(const RProp &net) : Backpropagation(net)
  {
    try {allocateSpace(net.nNodes);}
    catch (bad_alloc xa) {throw;}
    (*this) = net;
  }

  void RProp::operator=(const RProp &net)
  {
    DEBUG1("Attributing all values using assignment operator for RProp class");
    Backpropagation::operator=(net);

    deltaMax = net.deltaMax;
    deltaMin = net.deltaMin;
    incEta = net.incEta;
    decEta = net.decEta;
    initEta = net.initEta;

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


  RProp::RProp(const std::vector<unsigned> &nNodes, const std::vector<string> &trfFunc, const std::vector<bool> &usingBias, 
                      const REAL deltaMin = 1E-6, const REAL deltaMax = 50.0, const REAL initEta = 0.1,
                      const REAL incEta = 1.10, const REAL decEta = 0.5) : Backpropagation(nNodes, trfFunc, usingBias)
  {
    DEBUG1("Initializing the RProp class from scratch.");
    this->deltaMax = deltaMax;
    this->deltaMin = deltaMin;
    this->incEta = incEta;
    this->decEta = decEta;
    this->initEta = initEta;

    try {allocateSpace(nNodes);}
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


  void RProp::allocateSpace(const vector<unsigned> &nNodes)
  {
    DEBUG2("Allocating all the space that the RProp class will need.");
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
    DEBUG2("Releasing all memory allocated by RProp.");
    // Deallocating the delta bias matrix.
    releaseMatrix(prev_db);
    releaseMatrix(delta_b);

    // Deallocating the delta weights matrix.
    releaseMatrix(prev_dw);
    releaseMatrix(delta_w);
  }



  void RProp::updateWeights(const unsigned numEvents)
  {
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
  
  
  void RProp::showInfo() const
  {
    Backpropagation::showInfo();
    REPORT("TRAINING ALGORITHM INFORMATION");
    REPORT("Training algorithm: Resilient Backpropagation");
    REPORT("Maximum allowed learning rate value (deltaMax) = " << deltaMax);
    REPORT("Minimum allowed learning rate value (deltaMin) = " << deltaMin);
    REPORT("Learning rate increasing factor (incEta) = " << incEta);
    REPORT("Learning rate decreasing factor (decEta) = " << decEta);
    REPORT("Initial learning rate value (initEta) = " << initEta);
  }
}
