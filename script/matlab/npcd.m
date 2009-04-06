function [pcd, outNet, epoch, trnError, valError] = npcd(net, inTrn, inVal, deflation, numIterations, numThreads)
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

%Getting the desired network parameters.
[trnAlgo, useSP, numPCD, numNodes, trfFunc, trnParam] = getNetworkInfo(net);

%Initializing the output vectors.
pcd = [];
outNet = cell(1,numPCD);
epoch = cell(1,numPCD);
trnError = cell(1,numPCD);
valError = cell(1,numPCD);

%Extracting one PCD per iteration.
for i=1:numPCD,
  fprintf('Extracting PCD number %d\n', i);
  
  %Creating the neural network based on the PCD extraction method.
  if deflation,
    [net, inTrn, inVal] = defPCD(inTrn, inVal, pcd(), trnAlgo, useSP, numNodes, trfFunc, trnParam);
  else
    trnNet = stdPCD(pcd, trnAlgo, useSP, numNodes, trfFunc, trnParam);
  end
  
  %Doing the training.
  [trnNet, e, trnE, valE] = getBestTrain(trnNet, inTrn, inVal, numIterations, numThreads);
  
  %Saving the results.
  pcd = [pcd; trnNet.IW{1}(i,:)];
  outNet{i} = trnNet;
  epoch{i} = e;
  trnError{i} = trnE;
  valError{i} = valE;  
end


function net = stdPCD(pcd, trnAlgo, useSP, numNodes, trfFunc, trnParam)
  nPCD = size(pcd,1);
  numNodes(2) = nPCD + 1; %Increasing the number of nodes in the first hidden layers.
  net = newff2(numNodes, trfFunc, useSP, trnAlgo);
  trnNet.trainParam = trnParam;
  
  %Getting the PCDs extracted so far, and freezing them
  if (nPCD>=1),
    net.IW{1}(1:nPCD,:) = pcd;
    net.layers{1}.userdata.frozenNodes = (1:nPCD);
  end

  
function [net, inTrn, inVal] = defPCD(in_trn, in_val, pcd, trnAlgo, useSP, numNodes, trfFunc, trnParam)
  numNodes(2) = 1;
  net = newff2(numNodes, trfFunc, useSP, trnAlgo);
  trnNet.trainParam = trnParam;
  
  %Projecting the dataset on the PCD previously extracted.
  if ~isempty(pcd),
    inTrn = in_trn - in_trn * pcd * pcd;
    inVal = in_val - in_val * pcd * pcd;
  end
  
  
function [net, e, trnE, valE] = getBestTrain(net, inTrn, inVal, numIterations, numThreads)
  netVec = cell(1,numIterations);
  eVec = cell(1,numIterations);
  trnEVec = cell(1,numIterations);
  valEVec = cell(1,numIterations);
  effVec = zeros(1,numIterations);

  for i=1:numIterations,
    [auxNet, auxE, auxTrnE, auxValE] = ntrain(net, inTrn, inVal, numThreads);
    netVec{i} = auxNet;
    eVec{i} = auxE;
    trnEVec{i} = auxTrnE;
    valEVec{i} = auxValE;
    effVec(i) = mean(diag(genConfMatrix(nsim(net, inVal))));
  end
  
  %Getting the index where we achieve the highest mean efficiency.
  [val, idx] = max(effVec);

  net = netVec{idx};
  e = eVec{idx};
  trnE = trnEVec{idx};
  valE = valEVec{idx};
  

function [trnAlgo, useSP, numPCD, numNodes, trfFunc, trnParam] = getNetworkInfo(net)
  %Getting the network information regarding its topology

  %Taking the training algo.
  trnAlgo = net.trainFcn;

  %Checking whether or not to use SP or MSE network validation goal.
  useSP = net.userdata.useSP;

  %The number of PCDs to extract will be defined by the number of nodes in
  %the first hidden layer.
  numPCD = net.layers{1}.size;

  %Taking the other layer's size and training function.
  numNodes = [net.inputs{1}.size zeros(1,length(net.layers))];
  trfFunc = cell(1,length(net.layers));
  for i=1:length(net.layers),
    numNodes(i+1) = net.layers{i}.size;
    trfFunc{i} = net.layers{i}.transferFcn;
  end
  
  trnParam = net.trainParam;
  
