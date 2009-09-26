function [outNet, trnEvo, maxEfic] = numNodesEvolution(net, trn, val, tst, numIterations, minDiff)
%function [outNet, trnEvo, maxEfic] = numNodesEvolution(net, trn, val, tst, numIterations, minDiff)
%Variate the number of nodes in the first hidden layer and trains the
%resulting network numIteration times. The procedure wiil extract nodes 
%until the miimum relative diff minDiff [0,100] is reached during 3 
%consecutive trains
%

if (nargin < 5), numIterations = 5; end
if (nargin < 6), minDiff = 0.01; end

if (nargin > 6) || (nargin < 4),
  error('Invalid number of input arguments. See help.');
end

%Getting the desired network parameters.
[trnAlgo, maxNumNodes, numNodes, trfFunc, usingBias, trnParam] = getNetworkInfo(net);

%Initializing the output vectors.
outNet = cell(1,maxNumNodes);
trnEvo = cell(1,maxNumNodes);
maxEfic = zeros(1,maxNumNodes);

%Will count how many PCDs were actually extracted.
nNodes = 1;

%It is considered a failure if the max SP is less than minDiff the
%previous one. Then, if 'maxFail' failures occur, in a sequence, the
%analysis is aborted. But mxCount is reset to zero if, after a failure,
%the next extraction is successfull.
maxFail = 3;
mfCount = 0;
prevMaxSP = 0;
spDiff = 0;

%Extracting one PCD per iteration.
for i=1:maxNumNodes,
  nNodes = i;
  fprintf('Analysing %d (SP diff = %f)\n', nNodes, spDiff);
    
  trnNet = create_net(trnAlgo, numNodes, trfFunc, usingBias, trnParam, nNodes);
  
  %Doing the training.
  [nVec, idx] = trainMany(trnNet, trn, val, tst, numIterations);
  outNet{i} = nVec{idx}.net;
  trnEvo{i} = nVec{idx}.trnEvo;
  maxEfic(i) = nVec{idx}.sp;
  maxSP = 100*maxEfic(i);

  %If the SP increment is not above the minimum threshold, we initiate the
  %stopping countdown.
  spDiff = 100 * (maxSP-prevMaxSP) / maxSP;
  if (spDiff < minDiff)
    mfCount = mfCount + 1;
  else
    mfCount = 0; %Stopping the countdown for the moment.
  end
  
  if mfCount == maxFail,
    break; %We end the PCD extraction
  end
end

%Returning the num nodes actually validated..
outNet = outNet(1:nNodes);
trnEvo = trnEvo(1:nNodes);
maxEfic = maxEfic(1:nNodes);


function net = create_net(trnAlgo, numNodes, trfFunc, usingBias, trnParam, nNodes)
  numNodes.hidNodes(1) = nNodes;
  net = newff2(numNodes.inRange, numNodes.outRange, numNodes.hidNodes, trfFunc, trnAlgo);
  net.trainParam = trnParam;
  
  for i=1:length(net.layers),
    net.layers{i}.userdata.usingBias = usingBias(i);
  end
  

function [trnAlgo, maxNumPCD, numNodes, trfFunc, usingBias, trnParam] = getNetworkInfo(net)
  %Getting the network information regarding its topology

  %Taking the training algo.
  trnAlgo = net.trainFcn;

  %The maximum number of PCDs to be extracted is equal to the input size.
  maxNumPCD = net.inputs{1}.size;
  
  %Getting the input and output ranges.
  numNodes.inRange = net.inputs{1}.range;
  numNodes.outRange = net.outputs{length(net.outputs)}.range;

  %Taking the other layer's size and training function.
  numNodes.hidNodes = zeros(1,(length(net.layers)-1));
  trfFunc = cell(1,length(net.layers));
  usingBias = zeros(1,length(net.layers));
  for i=1:length(net.layers),
    if i < length(net.layers),
      numNodes.hidNodes(i) = net.layers{i}.size;
    end
    trfFunc{i} = net.layers{i}.transferFcn;
    usingBias(i) = net.layers{i}.userdata.usingBias;
  end
  
  trnParam = net.trainParam;
