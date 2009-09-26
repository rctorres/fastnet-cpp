function [pcd, outNet, trnEvo, efficVec] = npcd(net, inTrn, inVal, inTst, numIterations, minDiff)
%function [pcd, outNet, trnEvo, efficVec] = npcd(net, inTrn, inVal, inTst, numIterations, minDiff)
%Extracts the Principal Components of Discrimination (PCD).
%Input parameters are:
% net - The template neural netork to use. The number of PCDs to be
% extracted will be the same as the number of nodes in the first hidden
% layer.
% inTrn - The training data set, organized as a cell vector.
% inVal - The validating data set, organized as a cell vector.
% inTst - The testing data set, organized as a cell vector.
% numIterations - The number of times a neural network should be trained
% for extracting a given PCD. This is used to avoid local minima. For each
% PCD, the iteration which generated the best mean detection efficiency will
% provide the extracted PCD. Default is 10.
% minDiff - The minimum difference (in percentual value) in the SP for continuing extracting PCDs.
%
%The function returns:
% pcd - A matrix with the extracted PCDs.
% outNet - A cell vector containing the trained network structure obtained after
% each PCD extraction.
% trnEvo - A cell vector containing the training evolution data obtained during
% each PCD extraction.
% efficVec - a struct vector containing the mean and std of the SP efficiency obtained
% for each PCD extraction, considering the number of iterations performed.
%
%

if (nargin < 5), numIterations = 5; end
if (nargin < 6), minDiff = 0.01; end

if (nargin > 6) || (nargin < 4),
  error('Invalid number of input arguments. See help.');
end

%Getting the desired network parameters.
[trnAlgo, maxNumPCD, numNodes, trfFunc, usingBias, trnParam] = getNetworkInfo(net);

%If we have more than 2 layers (excluding the input), then we'll perform
%PCD extraction based on Caloba's rules, ensuring full PCD
%orthogonalization. Also, in this case, there must be no bias in the first
%hidded layer, and the activation function must be linear.
if length(trfFunc) > 2,
  multiLayer = true;
  usingBias(1) = false;
  trfFunc{1} = 'purelin';
else
  multiLayer = false;
end


%Initializing the output vectors.
pcd = [];
bias = [];
saveWeights = [];
outNet = cell(1,maxNumPCD);
trnEvo = cell(1,maxNumPCD);
meanEfic = zeros(1,maxNumPCD);
stdEfic = zeros(1,maxNumPCD);
maxEfic = zeros(1,maxNumPCD);

%Will count how many PCDs were actually extracted.
pcdExtracted = 1;

%It is considered a failure if the PCD max SP is less than minDiff the
%previous one. Then , if 'maxFail' failures occur, in a sequence, the PCD
%extraction is aborted. But mxCount is reset to zero if, after a failure,
%the next extraction is successfull.
maxFail = 3;
mfCount = 0;
prevMaxSP = 0;
spDiff = 0;

%Extracting one PCD per iteration.
for i=1:maxNumPCD,
  pcdExtracted = i;
  fprintf('Extracting PCD number %d (SP diff = %f)\n', pcdExtracted, spDiff);
  
  
  trnNet = stdPCD(pcd, bias, trnAlgo, numNodes, trfFunc, usingBias, trnParam);
  if (multiLayer && i>1),
    [trnNet, inTrn, inVal, inTst] = forceOrthogonalization(trnNet, inTrn, inVal, inTst);
  end
  
  %Doing the training.
  [nVec, idx] = trainMany(trnNet, inTrn, inVal, inTst, numIterations);
  outNet{i} = nVec{idx}.net;
  trnEvo{i} = nVec{idx}.trnEvo;
  maxEfic(i) = nVec{idx}.sp;
  maxSP = 100*nVec{idx}.sp;

  %Getting the mean and std val of the SP efficiencies obtained through the iterations.
  ef = zeros(1,numIterations);
  for j=1:numIterations,
    ef(j) = nVec{j}.sp;
  end
  meanEfic(i) = mean(ef);
  stdEfic(i) = std(ef);
  
  pcd = [pcd; outNet{i}.IW{1}(end,:)];
  bias = outNet{i}.b{1};

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
  
  % We move on to the next PCD.
  prevMaxSP = maxSP;
end

%Returning the PCDs actually extracted.
pcd = pcd(1:pcdExtracted,:);
outNet = outNet(1:pcdExtracted);
trnEvo = trnEvo(1:pcdExtracted);
efficVec.mean = meanEfic(1:pcdExtracted);
efficVec.std = stdEfic(1:pcdExtracted);
efficVec.max = maxEfic(1:pcdExtracted);


function net = stdPCD(pcd, bias, trnAlgo, numNodes, trfFunc, usingBias, trnParam)
  nPCD = size(pcd,1);
  numNodes.hidNodes(1) = nPCD + 1; %Increasing the number of nodes in the first hidden layers.
  net = newff2(numNodes.inRange, numNodes.outRange, numNodes.hidNodes, trfFunc, trnAlgo);
  net.trainParam = trnParam;
  
  for i=1:length(net.layers),
    net.layers{i}.userdata.usingBias = usingBias(i);
  end
  
  %Getting the PCDs extracted so far, and freezing them.
  if (nPCD>=1),
    net.IW{1}(1:nPCD,:) = pcd;
    net.b{1}(1:nPCD) = bias;
    net.layers{1}.userdata.frozenNodes = (1:nPCD);
  end
  
  
function [oNet, inTrn, inVal, inTst] = forceOrthogonalization(net, trn, val, tst)
  %If we  have already extracted a PCD, we remove
  % the information of the last PCD from the init values of the
  % new PCD to be extracted, and also from the input data.

  oNet = net;
    
  %Getting the last PCD extracted.
  W = net.IW{1}(end-1,:);
    
  %Removing the info related to the last PCD extracted.
  Nc = length(trn);
  inTrn = cell(1,Nc);
  inVal = cell(1,Nc);
  inTst = cell(1,Nc);
  for i=1:Nc,
    inTrn{i} = trn{i} - ( W' * (W * trn{i}) );
    inVal{i} = val{i} - ( W' * (W * val{i}) );
    inTst{i} = tst{i} - ( W' * (W * tst{i}) );
  end
  
  %Pointing the initial weights of the new PCD to the right direction.
  W_all = net.IW{1}(1:end-1,:);
  if ~isempty(W_all),
    sW = oNet.IW{1}(end,:);
    sW = sW - ( W_all' * (W_all * sW') )';
    oNet.IW{1}(end,:) = sW;
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
