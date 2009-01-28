#ifndef PATREC_H
#define PATREC_H

#include "fastnet/matlab/Training.h"

using namespace FastNet;

class PatternRecognition : public Training
{
private:
  vector<MatEvents*> inTrnList;
  vector<MatEvents*> inValList;
  vector<const REAL*> targList;
  vector< vector<REAL*>* > epochValOutputs;
  vector<unsigned> trnEpochList;
  bool useSP;


public:

  PatternRecognition(const mxArray *inTrn, const mxArray *inVal, const mxArray *epochSize, const bool usingSP) : Training()
  {
    if (mxGetN(inTrn) != mxGetN(inVal)) throw "Number of training and validating patterns are not equal";
    
    useSP = usingSP;
    if (useSP) bestGoal = 0.;
    const unsigned numPat = mxGetN(inTrn);
    for (unsigned i=0; i<numPat; i++)
    {
      //Getting the training data for each pattern.
      inTrnList.push_back(new MatEvents (mxGetCell(inTrn, i)));
      //Getting the val data for each pattern.
      inValList.push_back(new MatEvents (mxGetCell(inVal, i)));
      //Generating the desired output for each pattern for maximum sparsed outputs.
      REAL *target = new REAL [numPat];
      for (unsigned j=0; j<numPat; j++) target[j] = -1;
      target[i] = 1;
      //Saving the target in the list.
      targList.push_back(target);
      
      //Allocating space for the generated outputs...
      vector<REAL*> *aux = new vector<REAL*>;
      for (unsigned j=0; j<inValList[i]->getNumEvents(); j++) aux->push_back(new REAL [numPat]);
      epochValOutputs.push_back(aux);
    }

    if (!epochSize)
    {
      for (unsigned i=0; i<numPat; i++) trnEpochList.push_back(mxGetN(mxGetCell(inTrn, i)));
    }
    else
    {
      if ( (mxGetN(epochSize) != mxGetN(inTrn)) ) 
           throw "Number of patterns in the training events per epoch vector are not the same as the total number of patterns!";
      const double *numTrnEvEp = mxGetPr(epochSize);
      for (unsigned i=0; i<numPat; i++) trnEpochList.push_back(static_cast<unsigned>(numTrnEvEp[i]));
    }
  };

  virtual ~PatternRecognition()
  {
    for (unsigned i=0; i<inTrnList.size(); i++)
    {
      delete inTrnList[i];
      delete inValList[i];
      delete [] targList[i];
      for (unsigned j=0; j<epochValOutputs[i]->size(); j++) delete [] epochValOutputs[i]->at(j);
      delete epochValOutputs[i];
    }
  };

  /// Calculates the SP product.
  /**
  Calculates the SP product. This method will run through the dynamic range of the outputs,
  calculating the SP product in each lambda value. Returning, at the end, the maximum SP
  product obtained.
  @return The maximum SP value obtained.
  */
  REAL sp()
  {
    const REAL RESOLUTION = 0.001;
    unsigned TARG_SIGNAL, TARG_NOISE;
  
    //We consider that our signal has target output +1 and the noise, -1. So, the if below help us
    //figure out which class is the signal.
    if (targList[0][0] > targList[1][0]) // target[0] is our signal.
    {
      TARG_NOISE = 1;
      TARG_SIGNAL = 0;
    }
    else //target[1] is the signal.
    {
      TARG_NOISE = 0;
      TARG_SIGNAL = 1;
    }
  
    REAL maxSP = -1.;
    for (REAL pos = targList[TARG_NOISE][0]; pos < targList[TARG_SIGNAL][0]; pos += RESOLUTION)
    {
      REAL sigEffic = 0.;
      for (vector<REAL*>::const_iterator itr = epochValOutputs[TARG_SIGNAL]->begin(); itr != epochValOutputs[TARG_SIGNAL]->end(); itr++)
      {
        if ((*itr)[0] >= pos) sigEffic++;
      }
      sigEffic /= static_cast<REAL>(epochValOutputs[TARG_SIGNAL]->size());

      REAL noiseEffic = 0.;
      for (vector<REAL*>::const_iterator itr = epochValOutputs[TARG_NOISE]->begin(); itr != epochValOutputs[TARG_NOISE]->end(); itr++)
      {
        if ((*itr)[0] < pos) noiseEffic++;
      }
      noiseEffic /= static_cast<REAL>(epochValOutputs[TARG_NOISE]->size());

      //Using normalized SP calculation.
      const REAL sp = ((sigEffic + noiseEffic) / 2) * sqrt(sigEffic * noiseEffic);
      if (sp > maxSP) maxSP = sp;
    }
    return maxSP;
  };

  /// Applies the validating set of each pattern for the network's validation.
  /**
  This method takes the one or more pattern's validating events (input and targets) and presents them
  to the network. At the end, the mean training error is returned. Since it is a validating function,
  the network is not modified, and no updating weights values are calculated. This method only
  presents the validating sets and calculates the mean validating error obtained.
  @param[in] net the network class that the events will be presented to. The internal parameters
  of this class are not modified inside this method, since it is only a network validating process.
  @return The mean validating error obtained after the entire training set is presented to the network.
  */
  REAL valNetwork(NeuralNetwork *net)
  {
    REAL gbError = 0;
    const unsigned outSize = (*net)[net->getNumLayers()-1] * sizeof(REAL);
  
    for (unsigned pat=0; pat<inValList.size(); pat++)
    {
      //wFactor will allow each pattern to have the same relevance, despite the number of events it contains.
      const REAL wFactor = 1. / static_cast<REAL>(inValList.size() * inValList[pat]->getNumEvents());

      for (unsigned i=0; i<inValList[pat]->getNumEvents(); i++)
      {
        // Getting the next input and target pair.
        const REAL *out;
        const REAL *input = inValList[pat]->readEvent(i);
        gbError += ( wFactor * net->applySupervisedInput(input, targList[pat], out) );
        if (useSP) memcpy(epochValOutputs[pat]->at(i), out, outSize);
      }
    }

    return (useSP) ? sp() : gbError;
  };


  /// Applies the training set of each pattern for the network's training.
  /**
  This method takes the one or more patterns training events (input and targets) and presents them
  to the network, calculating the new mean (if batch training is being used) update values 
  after each input-output pair of each individual pattern is presented. At the end, the mean training error is returned.
  @param[in] net the network class that the events will be presented to. At the end,
  this class is modificated, as it will contain the mean values of \f$\Delta w\f$ and \f$\Delta b\f$ obtained
  after the entire training set has been presented, but the weights are not updated at the 
  end of this function. To actually update the weights, the user must call the proper
  class's method for that.
  @return The mean training error obtained after the entire training of each pattern set is presented to the network.
  */
  REAL trainNetwork(NeuralNetwork *net)
  {
    REAL gbError = 0;
    for(unsigned pat=0; pat<inTrnList.size(); pat++)
    {
      const REAL *targ = targList[pat];
    
      //wFactor will allow each pattern to have the same relevance, despite the number of events it contains.
      const REAL wFactor = 1. / static_cast<REAL>(inTrnList.size() * trnEpochList[pat]);
    
      for (unsigned i=0; i<trnEpochList[pat]; i++)
      {
        unsigned evIndex;
        const REAL *output;
        // Getting the next input and target pair.
        const REAL *input = inTrnList[pat]->readRandomEvent(evIndex);
        gbError += ( wFactor * net->applySupervisedInput(input, targ, output));
        //Calculating the weight and bias update values.
        net->calculateNewWeights(output, targ, pat);
      }
    }

    return gbError;  
  };
  
  vector<unsigned> getEpochSize() const {return trnEpochList;};
  
  void checkSizeMismatch(const NeuralNetwork *net) const
  {
    for (unsigned i=0; i<inTrnList.size(); i++)
    {
      if ( (inTrnList[i]->getEventSize() != (*net)[0]) || (inValList[i]->getEventSize() != (*net)[0]) )
        throw "Input training or validating data do not match the network input layer size!";
    }
  };


  void showInfo(const unsigned nEpochs) const
  {
    REPORT("TRAINING DATA INFORMATION (Pattern Recognition Optimized Network)");
    REPORT("Number of Epochs                    : " << nEpochs);
    REPORT("Using SP Stopping Criteria          : " << (useSP) ? "true" : "false");
    for (unsigned i=0; i<inTrnList.size(); i++)
    {
      REPORT("Information for pattern " << (i+1) << ":");
      REPORT("Number of training events per epoch : " << trnEpochList[i]);
      REPORT("Total number of training events     : " << inTrnList[i]->getNumEvents());
      REPORT("Total number of validating events      : " << inValList[i]->getNumEvents());
    }
  };

  bool isBestNetwork(const REAL currError)
  {
    if (useSP)
    {
      if (currError > bestGoal)
      {
        bestGoal = currError;
        return true;
      }
      return false;
    }
    
    //Otherwise we use the standard MSE method.
    return Training::isBestNetwork(currError);
  };

  virtual void showTrainingStatus(const unsigned epoch, const REAL trnError, const REAL valError)
  {
    if (useSP) {REPORT("Epoch " << setw(5) << epoch << ": mse (train) = " << trnError << "SP (val) = " << valError)}
    else Training::showTrainingStatus(epoch, trnError, valError);
  };

};

#endif