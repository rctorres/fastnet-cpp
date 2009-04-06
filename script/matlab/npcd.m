function [pcd, outNet, epoch, trnError, valError] = npcd(net, in_trn, in_val, deflation, numIterations, numThreads)
%HELP DA NPCD

if (nargin == 3),
  deflation = false;
  numIterations = 10;
  numThreads = 1;
elseif (nargin == 4),
  numIterations = 10;
  numThreads = 1;
elseif (nargin == 5),
  numThreads = 1;  
elseif (nargin > 6) || (nargin < 3),
  error('Invalid number of input arguments. See help.');
end

%Getting the network information regarding its topology

%Taking the training algo.
trnAlgo = net.trainFcn;

%Checking whether or not to use SP or MSE network validation goal.
useSP = net.userdata.useSP;

%The number of PCDs to extract will be defined by the number of nodes in
%the first hidden layer.
numPCD = net.layers{1}.size;

%Taking the other layer's size and training function.
numNodes = [net.inputs{1}.size];
trfFunc = [];
for i=1:length(net.layers),
  numNodes = [numNodes net.layers{i}.size];
  trfFunc = [trfFunc {net.layers{i}.transferFcn}];
end

outNet = [];
epoch = [];
trnError = [];
valError = [];

%Extracting one PCD per iteration.
for i=1:numPCD,
  fprintf('Extracting PCD number %d\n', i);
  %Creating the neural network
  numNodes(2) = i; %Increasing the number of nodes in the first hidden layers.
  trnNet = newff2(numNodes, trfFunc, useSP, trnAlgo);
  trnNet.trainParam = net.trainParam;
  
  %Getting the PCDs extracted so far, and freezing them
  if (i>1),
    trnNet.IW{1}(1:(i-1),:) = outNet(i-1).IW{1};
    trnNet.layers{1}.userdata.frozenNodes = [1:(i-1)];
  end
  
  %Initializing the variables wich will hold the result of this iteration.
  e = [];
  trnE = [];
  valE = [];

  %Doing the training.
  [trnNet, e, trnE, valE] = ntrain(trnNet, in_trn, in_val, numThreads);
    
  %Saving the results.
  outNet = [outNet trnNet];
  epoch = [epoch {e}];
  trnError = [trnError {trnE}];
  valError = [valError {valE}];  
end

pcd = outNet(end).IW{1};

