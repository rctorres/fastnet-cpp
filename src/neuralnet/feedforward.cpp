#include "fastnet/neuralnet/feedforward.h"

using namespace std;

namespace FastNet
{
    FeedForward::FeedForward(const FeedForward &net) : NeuralNetwork(net){}
    FeedForward::FeedForward(const mxArray *netStr) : NeuralNetwork(netStr){}
    NeuralNetwork *FeedForward::clone(){return new FeedForward(*this);}      
    FeedForward::~FeedForward() {}
}
