/** 
@file  backpropagation.cpp
@brief The BackPropagation class definition.
*/

#include <vector>
#include <string>
#include <cstdlib>

#include "fastnet/neuralnet/backpropagation.h"
#include "fastnet/reporter/Reporter.h"

namespace FastNet
{
  Backpropagation::Backpropagation(const Backpropagation &net) : NeuralNetwork(net)
  { 
    wFactor.assign(net.wFactor.begin(), net.wFactor.end());
    learningRate = net.learningRate;
    decFactor = net.decFactor;
 
    try {allocateSpace();}
    catch (bad_alloc xa) {throw;}

    for (unsigned i=0; i<(nNodes.size() - 1); i++)
    {
      memcpy(db[i], net.db[i], nNodes[i+1]*sizeof(REAL));
      memcpy(sigma[i], net.sigma[i], nNodes[i+1]*sizeof(REAL));
      for (unsigned j=0; j<nNodes[i+1]; j++) memcpy(dw[i][j], net.dw[i][j], nNodes[i]*sizeof(REAL));
    }
  }
  

  Backpropagation::Backpropagation(const mxArray *netStr, const vector<unsigned> &nEvPat) : NeuralNetwork(netStr)
  {
    DEBUG1("Initializing the Backpropagation class from a Matlab Network structure.");

    //Calculating the weightening factors.
    if (nEvPat.size() == 1) createWeighteningValues(nEvPat[0]);
    else if (nEvPat.size() > 1) createWeighteningValues(nEvPat);
    else throw "You must provide the number of events to be used for each epoch!.";

    //We first test whether the values exists, otherwise, we use default ones.
    const mxArray *trnParam =  mxGetField(netStr, 0, "trainParam");
    if (mxGetField(trnParam, 0, "lr")) this->learningRate = static_cast<REAL>(abs(mxGetScalar(mxGetField(trnParam, 0, "lr"))));
    else this->learningRate = 0.05;
    if (mxGetField(trnParam, 0, "decFactor")) this->decFactor = static_cast<REAL>(abs(mxGetScalar(mxGetField(trnParam, 0, "decFactor"))));
    else this->decFactor = 1;

    try {allocateSpace();}
    catch (bad_alloc xa) {throw;}

    //Initializing dw and db.
    for (unsigned i=0; i<(nNodes.size() - 1); i++)
    {
      for (unsigned j=0; j<nNodes[i+1]; j++) 
      {
        this->db[i][j] = 0.;
        this->sigma[i][j] = 0.;
        for (unsigned k=0; k<nNodes[i]; k++) this->dw[i][j][k] = 0.;
      }
    }
  }


  void Backpropagation::allocateSpace()
  {
    const unsigned size = nNodes.size() - 1;
    try
    {
      db = new REAL* [size];
      sigma = new REAL* [size];
      dw = new REAL** [size];
      for (unsigned i=0; i<size; i++)
      {
        db[i] = new REAL [nNodes[i+1]];
        sigma[i] = new REAL [nNodes[i+1]];
        dw[i] = new REAL* [nNodes[i+1]];
        for (unsigned j=0; j<nNodes[i+1]; j++) dw[i][j] = new REAL [nNodes[i]];
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
  }

  void Backpropagation::retropropagateError(const REAL *output, const REAL *target)
  {
    const unsigned size = nNodes.size() - 1;

    for (unsigned i=activeNodes[size].init; i<activeNodes[size].end; i++) sigma[size-1][i] = (target[i] - output[i]) * CALL_TRF_FUNC(trfFunc[size-1])(output[i], true);

    //Retropropagating the error.
    for (int i=(size-2); i>=0; i--)
    {
      for (unsigned j=activeNodes[i+1].init; j<activeNodes[i+1].end; j++)
      {
        sigma[i][j] = 0;

        for (unsigned k=activeNodes[i+2].init; k<activeNodes[i+2].end; k++)
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
    const REAL val = wFactor[0];

    retropropagateError(output, target);

    //Accumulating the deltas.
    for (unsigned i=0; i<size; i++)
    {
      for (unsigned j=activeNodes[(i+1)].init; j<activeNodes[(i+1)].end; j++)
      {
        for (unsigned k=activeNodes[i].init; k<activeNodes[i].end; k++)
        {
          dw[i][j][k] += (val * sigma[i][j] * layerOutputs[i][k]);
        }

        db[i][j] += (val * sigma[i][j]);
      }
    }
  }

  void Backpropagation::calculateNewWeights(const REAL *output, const REAL *target, const unsigned patIdx)
  {
    const unsigned size = nNodes.size() - 1;
    const REAL val = wFactor[patIdx];

    retropropagateError(output, target);

    //Accumulating the deltas.
    for (unsigned i=0; i<size; i++)
    {
      for (unsigned j=activeNodes[(i+1)].init; j<activeNodes[(i+1)].end; j++)
      {
        for (unsigned k=activeNodes[i].init; k<activeNodes[i].end; k++)
        {
          dw[i][j][k] += (val * sigma[i][j] * layerOutputs[i][k]);
        }

        db[i][j] += (val * sigma[i][j]);
      }
    }
  }
  
  void Backpropagation::updateWeights()
  {    
    for (unsigned i=0; i<(nNodes.size()-1); i++)
    {
      for (unsigned j=activeNodes[i+1].init; j<activeNodes[(i+1)].end; j++)
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
            weights[i][j][k] += (learningRate * dw[i][j][k]);
            dw[i][j][k] = 0;
          }

          if (usingBias[i])
          {
            bias[i][j] += (learningRate * db[i][j]);
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

  void Backpropagation::createWeighteningValues(const unsigned nPat)
  {
    wFactor.clear();
    wFactor.push_back( 1. / static_cast<REAL>(nPat) );
    DEBUG2("Number of events = " << nPat << ". Weightening value = " << wFactor[0]);
  }

  void Backpropagation::createWeighteningValues(const vector<unsigned> &nPat)
  {
    wFactor.clear();
    for (vector<unsigned>::const_iterator itr = nPat.begin(); itr != nPat.end(); itr++)
    {
      const REAL val = 1. / (static_cast<REAL>( nPat.size() * (*itr) ));
      wFactor.push_back(val);
      DEBUG2("Number of events = " << *itr << ". Weightening value = " << val);
    }
  }

  void Backpropagation::showInfo(ostream &str) const
  {
    NeuralNetwork::showInfo(str);
    str << "TRAINING ALGORITHM INFORMATION:" << endl;
    str << "Training algorithm : Gradient Descent" << endl;
    str << "Learning rate      : " << learningRate << endl;
    str << "Decreasing factor  : " << decFactor << endl;
  }
}
