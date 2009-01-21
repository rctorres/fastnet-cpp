/** 
@file  rprop.h
@brief The Resilient BackPropagation (RProp) class declaration.
*/
 
#ifndef RPROP_H
#define RPROP_H

#include <vector>

#include "fastnet/neuralnet/backpropagation.h"
#include "fastnet/defines.h"

using namespace std;


namespace FastNet
{
  /** 
  @brief    The Resilient BackPropagation (RProp) training class.
  @author    Rodrigo Coura Torres (torres@lps.ufrj.br)
  @version  1.0
  @date    14/11/2004

  This class implements the resilient backpropagation training algorithm.
  This algorithm is based only in the direction of the derivative. The 
  weights are updated by an adaptive value, so this class does not need an
  update factor.
  The class can perform either online and batch training, since the
  instant gradients are automatically accumulated every time the
  new weights values are calculated and also the class keeps control
  of how many inputs have been applied to the network, so the instant gradient
  is used at the update weights phase if only one input was applied, and the mean
  gradient will be used if multiple inputs were presented to the network. The class
  also automatically resets the accumulated values after an epoch, preparing itself
  for the next epoch, so the user just have to use the methods, without worring
  about internal control.
  */
  class RProp : public Backpropagation
  {
    protected:
      //Class attributes.

      /// The maximum allowed learning rate value.
      /**
       Since the upate value can be increased or decreased, this value
       specifies the maximum accepted value for the update factor.
      */
      REAL deltaMax;

      /// The minimum allowed learning rate value.
      /**
       Since the upate value can be increased or decreased, this value
       specifies the minimum accepted value for the update factor.
      */
      REAL deltaMin;

      /// Specifies the increase factor for the learning rate.
      /**
       If the learning rate must be increased, this factor specifies by
       how much the learning rate will be increased (the learning rate value
       is multiplyed by this attribute value).
      */
      REAL incEta;
      
      /// Specifies the decrease factor for the learning rate.
      /**
       If the learning rate must be decreased, this factor specifies by
       how much the learning rate will be decreased (the learning rate value
       is multiplyed by this attribute value).
      */
      REAL decEta;

      /// The initial learning rate value.
      /**
       This attribute stores the initial learning rate value. After the begining
       of the training, the current learning rate will be changed according
       to the Resilient Backpropagation algorithm.
      */
      REAL initEta;

      /// Stores the delta weights values of the previous training epoch.
      /**
       Since the RProp algorithm must know the previous delta weight values,
       in order to determine if the learning rate must be increased or decreased,
       this pointer holds a copy of the delta weights values calculated in the last epoch.
      */
      REAL ***prev_dw;

      /// Stores the delta biases values of the previous training epoch.
      /**
       Since the RProp algorithm must know the previous delta biases values,
       in order to determine if the learning rate must be increased or decreased,
       this pointer holds a copy of the delta biases values calculated in the last epoch.
      */
      REAL **prev_db;
      
      
      /// The learning rate value for each weight.
      /**
       The speeed of this algorithm relies on the fact that it can have
       an specific learning rate value for each weight. So, this pointer
       contains the learning rate values that will be used in each weight.
      */
      REAL ***delta_w;
      
      /// The learning rate value for each bias.
      /**
       The speeed of this algorithm relies on the fact that it can have
       an specific learning rate value for each bias. So, this pointer
       contains the learning rate values that will be used in each bias.
      */
      REAL **delta_b;

      //Inline methods.
      
      /// Gets the smaller of two numbers.
      /**
       This method takes two numbers and returns the smallest of them.
       @param[in] v1 The first number.
       @param[in] v2 The second number.
       @return v1 if v1 < v2, v2 otherwise.
      */
      REAL min(REAL v1, REAL v2) const {return ((v1 < v2) ? v1 : v2);}
    
      
      /// Gets the largest of two numbers.
      /**
       This method takes two numbers and returns the bigest of them.
       @param[in] v1 The first number.
       @param[in] v2 The second number.
       @return v1 if v1 > v2, v2 otherwise.
      */
      REAL max(REAL v1, REAL v2) const {return ((v1 > v2) ? v1 : v2);}
      
      
      /// Gets the sign of a number.
      /**
       This function gets the sign of a number.
       @param[in] val The number which the sign we want to know.
       @return 1 if val > 0, -1 if val <0, 0 if val = 0.
      */
      REAL sign(REAL val) const {if (val > 0) return 1; else if (val < 0) return -1; else return 0;}

      //Standart methods.


      /// Applies the RProp update weight algorithm.
      /**
       This method updates each weight and bias by applying the
       RProp algorithm for weights update. It also saves the current weight
       value for usage in the next epoch. It also resets the weight and biases
       values to zero, so they are ready to be used in the next epoch. This method
       works in a single weight or bias value. So, it will be called as many times
       as the number of weights and bias in the network.
       @param delta The learning rate value for the weight or bias.
       @param d The current delta weight (or bias) value.
       @param prev_d The previous delta weight (or bias) value.
       @param w The weight (or bias) value.
      */
      void updateW(REAL &delta, REAL &d, REAL &prev_d, REAL &w);

    public:
      //Base class virtual methods overrided.

      /// Update the weights and bias matrices.
      /**
       Update the bias and weight matrices. It uses the mean
       gradient sign. It is also prepared to work with nodes activation
       and frozen nodes.
       @see FastNet::NeuralNetwork#updateWeights()
      */
      void updateWeights();
      

      //Standart methods.

      /// Class constructor.
      /**
       This constructor allocates all the memory needed by the class. It
       also initializes the weights and biases learning rate values with
       the initial value and also the delta weights and biases values with zero.
       @param[in] nodesDist a vector containig the number of nodes in each layer (including the input layer).
       @param[in] trfFunction a vector containig the type of transfer function in each hidden and output layer.
       @throw bad_alloc in case of error during memory allocation.
      */
      RProp(vector<unsigned> nodesDist, vector<string> trfFunction);

      ///Copy constructor
      /**This constructor should be used to create a new network which is an exactly copy 
        of another network.
        @param[in] net The network that we will copy the parameters from.
      */
      RProp(const RProp &net);
      
      /// Class destructor.
      /**
       Releases all the dynamically allocated memory used by the class,
       so the user does not need to worry about dynamically allocated memory..
      */
      virtual ~RProp();
      
      
      /// Gives the neural network information.
      /**
       This method sends to a stream text information about the neural
       network. This method sould complement the information given by the
       base class.
       @param[in] str The stream where the information will be written to.
       @see FastNet::NeuralNetwork#showInfo 
      */
      virtual void showInfo(ostream &str) const;
  };
}

#endif
