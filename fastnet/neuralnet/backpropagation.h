/** 
@file  backpropagation.h
@brief The BackPropagation class declaration.
*/

 
#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include <vector>
#include <cstring>

#include <mex.h>

#include "fastnet/sys/defines.h"
#include "fastnet/neuralnet/neuralnetwork.h"

using namespace std;


namespace FastNet
{
  /** 
  @brief    The BackPropagation training class.
  @author    Rodrigo Coura Torres (torres@lps.ufrj.br)
  @version  1.0
  @date    14/11/2004

  This class implements the backpropagation training algorithm.
  it can perform either online and batch training, since the
  instant gradients are automatically accumulated every time the
  new weights values are calculated and also the class keeps control
  of how many inputs have been applied to the network, so the instant gradient
  is used at the update weights phase if only one input was applied, and the mean
  gradient will be used if multiple inputs were presented to the network. The class
  also automatically resets the accumulated values after an epoch, preparing itself
  for the next epoch, so the user just have to use the methods, without worring
  about internal control.
  */
  class Backpropagation : public NeuralNetwork 
  {
    protected:
      //Class attributes.
      
      /// The learning rate value to be used during the training process.
      REAL learningRate;    
      
      /// The decreasing factor to be applied to the learning rate value, after each epoch.
      /**
        This variable is used to decrease the learning rate value, in order to
        avoid oscilations arround the minumum error. The value stored in this variable
        should be \f$0 < df \leq 1\f$, so that, if the decrease factor is 0.98, for instance, after each epoch,
        the learning rate will be decreased by 2% of its previously value.
      */
      REAL decFactor;      
      
      /// Contains all the gradient of each node.
      /**
        The values stored in this matrix are the gradient calculated during
        the backpropagation phase of the algorithm. It will be used to calculate
        the update weight values. This variable is dynamically allocated by the class
        and automatically released at the end.
      */
      REAL **sigma;
      
      /// Contains the delta weight values.
      /**
        Contains the update values for each weight. This variable is dynamically allocated by the class
        and automatically released at the end. 
      */
      REAL ***dw;

      /// Contains the delta biases values.
      /**
        Contains the update values for each bias. This variable is dynamically allocated by the class
        and automatically released at the end. 
      */
      REAL **db;
      
      /// Hold the weightening values for each class.
      /*
      This vector contains the weightening values to be used when calculating the gradients. The
      weightening factors allow that, in case of a pattern rcognition network, that each class
      has the same relevance, no matter how many events it has.
      @see Backpropagation::createWeighteningValues
      */
      vector<REAL> wFactor;
      
      /// The saving weights matrix.
      /**
       Stores the weights values if desired by the user. The dimensions (w[x][y][z])are:
        - x: the layer index (where 0 is the first hidden layer).
        - y: the index of the node in layer x.
        - z: the index of the node in layer x-1.
      */
      REAL ***savedW;
      

      /// Stores the entwork bias
      /**
       Stores the weights values if desired by the user. The dimensions (b[x][y]) are:
        - x: the layer index (where 0 is the first hidden layer).
        - y: the index of the node in layer x.
      */
      REAL **savedB;

      /// Tells which nodes are frozen.
      /**
       This matrix store bolean values that tells if the corresponding
       node is frozen or not. If a node is frozen, the weights connected to
       its input will not be changed. Otherwise, the node works normally.
       By default, the class automatically makes all nodes unfrozen. If the
       user wants to freeze an specific node he must do that by means
       of calling the specific method for this purpose.
       The dimensions of this matrix are (frozenNode[x][y]):
        - x: the layer index (where 0 is the first hidden layer).
        - y: The node within layer x.
       @see FastNet::NeuralNetwork#setFreeze
      */
      bool **frozenNode;

      
      /// Retropropagates the error through the neural network.
      /**
       This method generatesthe error at the output of the network, comparing
       the output generated with the target value and retropropagates the 
       generated error through the network, in order to calculate the sigma
       values in for each node.
       @param[in] output The output genarated by the neural network.
       @param[in] target The desired (target) output value.
      */
      virtual void retropropagateError(const REAL *output, const REAL *target);

      //Dynamically allocates all the memory we need.
      /**
      This function will take the nNodes vector ans will allocate all the memory that must be
      dynamically allocated.
      */
      virtual void allocateSpace(const vector<unsigned> &nNodes);
      
      ///Create the weighting values for dw and db for the pattern recognition optimized training.
      /**
      When calculating the new gradients, they must be weightened by the number of events in each
      epochs, so that each patterm will have the same relevance when calculating the
      new gradients, no matter how many events it has. This method calculates the weightening values
      (\f$ wf \f$) in advance, to improve the training speed. The method considers the case when we have
      separate classes (pattern recognition). The weighting values are calculated
      according to the following rule:
      \f$ wf_i = \frac{\prod_{j=1/j \neq i}^{N_c}}{N_c \times \prod_{j=1}^{Nc} N_j} = \frac{1}{N_c \times N_i}\f$,
      where \f$ N_c \f$ is the number of patterns, and \f$ N_j \f$ is the total number of events,
      for the j-th pattern, to be presented to the network, per epoch, during the training.
      @param[in] nPat A vector containing the number of events to be used per epoch, for each pattern.
      */
      virtual void createWeighteningValues(const vector<unsigned> &nPat);

      ///Create the weighting values for dw and db for a normal network training.
      /**
      When calculating the new gradients, they must be weightened by the number of events to calculate
      the mean gradient. The method considers the stardart case, where you have simply a
      set of inputs and targets, not separated by classes. For this case, the weightening factor will
      be simply \f$ wf = \frac{1}{N}\f$, where \f$ N \f$ is the total number of events applied, per
      epoch, to the neural network.
      @param[in] nPat The number of events to be used per epoch.
      */      
      virtual void createWeighteningValues(const unsigned nPat);

    public:
      //Class virtual methods.

      ///Adds the gradient of another network to the calling network.
      /**
      This emthod will take the gradient info from the passed network, and add to
      the gradients of the calling network. This function is mainly for using when
      applying sample parallelisum for the training, using multi-threads, for instance.
      @param[in] net The network from where to get the gradients from.
      */
      virtual void addToGradient(const Backpropagation &net);
      
      /// Sets the freeze/unfreeze status of an specific node.
      /**
       Thos methods sets the freeze status of an specific node. If a node is frozen,
       the weights connected to its input are not changed.
       @param[in] layer The layer where the node to be set as frozen/unfrozen is (where 0 is the first hidden layer).
       @param[in] node The index of the node.
       @param[in] freezed If true, the node is set as freezed, otherwise it is set as unfreezed.
      */
      void setFrozen(unsigned layer, unsigned node, bool frozen)
      {
        frozenNode[layer][node] = frozen;
      };

      
      /// Sets the frozen/unfrozen status of an entire layer.
      /**
       Thos methods sets the freeze status for all the nodes in a specific layer. 
       If a node is frozen, the weights connected to its input are not changed.
       @param[in] layer The layer where the nodes to be set as freeze/unfreeze are (where 0 is the first hidden layer).
       @param[in] freezed If true, the node is set as freezed, otherwise it is set as unfreezed.
      */
      void setFrozen(unsigned layer, bool frozen){for (unsigned i=0; i<nNodes[layer+1]; i++) setFrozen(layer, i, frozen);};
      
      
      /// Tells if a node is frozen or not.
      /**
       param[in] layer The layer where the node frozen status is (where 0 is the first hidden layer).
       param[in] node The index of the node in the layer.
       @return True if the node is frozen, false otherwise.
      */
      bool isFrozen(unsigned layer, unsigned node) const {return frozenNode[layer][node];};
      
      
      /// Tells if an entire layer is frozen.
      /**
       This method checks if all the active nodes of an specific layer are frozen or not.
       param[in] layer The layer to be checked (where 0 is the first hidden layer).
       @return True if all nodes are frozen, false if one or more nodes are unfrozen.
      */
      bool isFrozen(unsigned layer) const;


      /// Defrost all nodes in the network.
      /**
       This method goes through the network and unfrost every node in each.
      */
      void defrostAll(){for (unsigned i=0; i<(nNodes.size()-1); i++) setFrozen(i, false);};

      /// Propagates and input event and calculates the MSE error obtained by comparing to a target output.
      /**
       This method should be used only in supervised
       training algorithms. It propagates an input through the network, and, after 
       comparing the output generated with the desired (target) output, calculates
       the MSE error obtained by the relation below:
       
       \f$ e = \frac{1}{N} \sum\limits_{i=0}^{N-1} \left ( t[i] - o[i] \right )^2 \f$

       where:
        - N is the number of nodes in the output layer.
        - o[i] is the output generated by the network at the ith node.
        - t[i] is the desired output to the ith node.

       @param[in] input The vector containing the input to be presented to the network.
       @param[in] target The vector containing the desired output (target) of the network.
       @param[out] output This pointer will point to the output generated by the network. It must
       not be deallocated after use. The class will automatically release the memory used by
       this vector.
       @return The MSE error calculated.
      */
      virtual REAL applySupervisedInput(const REAL *input, const REAL *target, const REAL* &output);


      /// Flush weights from memory to a Matlab variable.
      /**
       Since this class, in order to optimize speed, saves the
       weights and bias values into memory, at the end, if the user wants
       to save the final values, this method must be called. It will
       save the weights and biases values stored in the memory buffer in a matlab variable.
       So, this method can only be used after the writeWeights has been called at least once.
       @param[out] outNet The matlab network structure to where the weights and biases will be saved to.
      */
      void flushBestTrainWeights(mxArray *outNet) const;

      /// Writes the weights in a memory buffer.
      /**
       To improve speed during training, where the weights values change a lot,
       this function stores these values in a memory buffer, without loss
       of performance. But this function DOES NOT stores those values in a non-volatile
       environment, so these values are lost after the training is finished. In order to
       actually save the weights and biases for posterior use in matlab, you must call, after the training is
       done, the flushBestTrainWeights method.
       @see FastNet::MatNetData#flushBestTrainWeights for information on how the save 
       the weights and biases for posterior use in matlab.
      */
      void saveBestTrain()
      {
        for (unsigned i=0; i<(nNodes.size()-1); i++)
        {
          memcpy(savedB[i], bias[i], nNodes[(i+1)]*sizeof(REAL));
          for (unsigned j=0; j<nNodes[(i+1)]; j++) memcpy(savedW[i][j], weights[i][j], nNodes[i]*sizeof(REAL));
        }
      };


      /// Ramdomly initializes the weight and bias values.
      /**
       This method ramdomly initializes the weight and bias values of a neural
       network.
       @param[in] initWeightRange the weights's range, so that -initWeightRange \f$\leq\f$ w \f$\leq\f$ initWeightRange.
      */
      virtual void initWeights(REAL initWeightRange);

      /// Calculates the new weight values.
      /**
       This method retropropagates the error through the network
       and accumulates the local gradients of each weight and bias,
       for batch training. It uses the previously calculated weightened factor, so we can get the
       correct mean gradient.
       @param[in] output The output generated by the network after the feedforward process.
       @param[in] target The desired (target) output.
      */
      virtual void calculateNewWeights(const REAL *output, const REAL *target);

      /// Calculates the new weight values taking into count the number of events that are being presented.
      /**
       This method retropropagates the error through the network
       and accumulates the local gradients of each weight and bias,
       for batch training. This method take into count the a priori number of events
       that will be presented to the neural network, so it can be used when
       a discrimination network is being trained and you have different number
       of events for each pattern, since the a priori number of events is used
       so each class can have the same relevance during training, no matter
       the number of events in each class.
       @param[in] output The output generated by the network after the feedforward process.
       @param[in] target The desired (target) output.
       @param[in] patIdx the index (starting in 0) of the pattern being applied, so we can get the
       corresponding weightening factor.
       */
      virtual void calculateNewWeights(const REAL *output, const REAL *target, const unsigned patIdx);

      /// Updates the weight and biases matrices.
      /**
       Update the bias and weight matrices. It uses the mean
       gradient calculated each time the method calculateNewWeights is called.
       So, if you don't want to use batch training, you should call the calculateNewWeights
       and then the updateWeights method every time a new training input is applied to the network.
       This method also sets dw, db and trnEventCounter to zero, in order to start a new epoch. 
       So, the class automatically manages the batch/non-batch training process and the user
       don't need to worry about that.
      */
      virtual void updateWeights();

      ///Copy constructor
      /**This constructor should be used to create a new network which is an exactly copy 
        of another network.
        @param[in] net The network that we will copy the parameters from.
      */
      Backpropagation(const Backpropagation &net);


      /// Constructor taking the parameters for a matlab net structure.
      /**
      This constructor should be called when the network parameters are stored in a matlab
      network structure.
      @param[in] netStr The Matlab network structure as returned by newff.
      @param[in] nEvPat A vector containing the number of events to be used for each pattern.
      If you are not using a pattern recognition optimized network, the vector must contain only
      one value, corresponding to the total number of events to be presented for each epoch.
      */
      Backpropagation(const mxArray *netStr, const vector<unsigned> &nEvPat);

      /// Returns a clone of the object.
      /**
      Returns a clone of the calling object. The clone is dynamically allocated,
      so it must be released with delete at the end of its use.
      @return A dynamically allocated clone of the calling object.
      */
      virtual NeuralNetwork *clone(){return new Backpropagation(*this);} 
      
      /// Class destructor.
      /**
       Releases all the dynamically allocated memory used by the class.
      */
      virtual ~Backpropagation();
      
      
      /// Gives the neural network information.
      /**
       This method prints information about the neural
       network. This method sould complement the information given by the
       base class.
       @see FastNet::NeuralNetwork#showInfo 
      */
      virtual void showInfo() const;

      //Copy the status from the passing network.
      /**
        This method will make a deep copy of all attributes from the passing network,
        making them exactly equal. This method <b>does not</b> allocate any memory for
        the calling object. The space for weights and bias info must have been previously created.
        @param[in] net The network from where to copy the data from.
      */
      virtual void operator=(const Backpropagation &net);
  };
}

#endif
