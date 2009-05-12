function [pcd, outNet, epoch, trnError, valError, efficVec] = npcd(net, inTrn, inVal, inTst, deflation, numIterations)
%function [pcd, outNet, epoch, trnError, valError, efficVec] = npcd(net, inTrn, inVal, inTst, deflation, numIterations)
%Extracts the Principal Components of Discrimination (PCD).
%Input parameters are:
% net - The template neural netork to use. The number of PCDs to be
% extracted will be the same as the number of nodes in the first hidden
% layer.
% inTrn - The training data set, organized as a cell vector.
% inVal - The validating data set, organized as a cell vector.
% inTst - The testing data set, organized as a cell vector.
% deflation - if true, the PCDs will be extracted by the deflation method.
% In this case, the returned neural network MUST NOT be used as a
% discriminator. Deflation should be used ONLY if you want just to extract
% the PCDs, and not develop the classifier at the same time. Default is
% false.
% numIterations - The number of times a neural network should be trained
% for extracting a given PCD. This is used to avoid local minima. For each
% PCD, the iteration which generated the best mean detection efficiency will
% provide the extracted PCD. Default is 10.
%
%The function returns:
% pcd - A matrix with the extracted PCDs.
% outNet - A cell vector containing the trained network structure obtained after
% each PCD extraction.
% epoch - A cell vector containing the epochs evolution obtained during
% each PCD extraction.
% trnError - A cell vector containing the training error evolution obtained
% during each PCD extraction.
% valError - A cell vector containing the validation error evolution
% obtained during each PCD extraction.
% efficVec - a struct vector containing the mean and std of the SP efficiency obtained
% for each PCD extraction, considering the number of iterations performed.
%


if (nargin == 4),
  deflation = false;
  numIterations = 5;
elseif (nargin == 5),
  numIterations = 5;
elseif (nargin > 6) || (nargin < 4),
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
epoch = cell(1,maxNumPCD);
trnError = cell(1,maxNumPCD);
valError = cell(1,maxNumPCD);
meanEfic = zeros(1,maxNumPCD);
stdEfic = zeros(1,maxNumPCD);
maxEfic = zeros(1,maxNumPCD);

%Will count how many PCDs were actually extracted.
pcdExtracted = 1;

%It is considered a failure if the PCD max SP is less than minDiff the
%previous one. Then , if 'maxFail' failures occur, in a sequence, the PCD
%extraction is aborted. But mxCount is reset to zero if, after a failure,
%the next extraction is successfull.
minDiff = 0.0003;
maxFail = 3;
mfCount = 0;
prevMeanSP = 0;
spDiff = 0;

%Extracting one PCD per iteration.
for i=1:maxNumPCD,
  pcdExtracted = i;
  fprintf('Extracting PCD number %d (SP diff = %f)\n', pcdExtracted, spDiff);
  
  
  %Creating the neural network based on the PCD extraction method.
  if deflation,
    [trnNet, inTrn, inVal] = defPCD(inTrn, inVal, pcd, trnAlgo, numNodes, trfFunc, usingBias, trnParam);
  else
    trnNet = stdPCD(pcd, bias, trnAlgo, numNodes, trfFunc, usingBias, trnParam);
    if (multiLayer),
      [trnNet, inTrn, inVal, inTst, saveWeights] = forceOrthogonalization(trnNet, inTrn, inVal, inTst, saveWeights);
    end
  end
  
  %Doing the training.
  [nVec, idx] = trainMany(trnNet, inTrn, inVal, inTst, numIterations, multiLayer);
  outNet{i} = nVec{idx}.net;
  epoch{i} = nVec{idx}.epoch;
  trnError{i} = nVec{idx}.trnError;
  valError{i} = nVec{idx}.valError;
  maxEfic(i) = nVec{idx}.sp;

  %Getting the mean and std val of the SP efficiencies obtained through the iterations.
  ef = zeros(1,numIterations);
  for j=1:numIterations,
    ef(j) = nVec{j}.sp;
  end
  meanEfic(i) = mean(ef);
  meanSP = meanEfic(i);
  stdEfic(i) = std(ef);
  
  pcd = [pcd; outNet{i}.IW{1}(end,:)];
  bias = outNet{i}.b{1};
  
  %If the SP increment is not above the minimum threshold, we initiate the
  %stopping countdown.
  spDiff = meanSP-prevMeanSP;
  if (spDiff < minDiff)
    mfCount = mfCount + 1;
  else
    mfCount = 0; %Stopping the countdown for the moment.
  end
  
  if mfCount == maxFail,
    break; %We end the PCD extraction
  end
  
  % We move on to the next PCD.
  prevMeanSP = meanSP;
end

%Returning the PCDs actually extracted.
pcd = pcd(1:pcdExtracted,:);
outNet = outNet(1:pcdExtracted);
epoch = epoch(1:pcdExtracted);
trnError = trnError(1:pcdExtracted);
valError = valError(1:pcdExtracted);
efficVec.mean = meanEfic(1:pcdExtracted);
efficVec.std = stdEfic(1:pcdExtracted);
efficVec.max = maxEfic(1:pcdExtracted);


function net = stdPCD(pcd, bias, trnAlgo, numNodes, trfFunc, usingBias, trnParam)
  nPCD = size(pcd,1);
  numNodes(2) = nPCD + 1; %Increasing the number of nodes in the first hidden layers.
  net = newff2(numNodes, trfFunc, trnAlgo);
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

  
  
function [net, inTrn, inVal] = defPCD(in_trn, in_val, pcd, trnAlgo, numNodes, trfFunc, usingBias, trnParam)
  numNodes(2) = 1;
  net = newff2(numNodes, trfFunc, trnAlgo);
  net.trainParam = trnParam;

  for i=1:length(net.layers),
    net.layers{i}.userdata.usingBias = usingBias(i);
  end
  
  %Projecting the dataset on the PCD previously extracted.
  if ~isempty(pcd),
    W = pcd(end,:);
    nClasses = length(in_trn);
    inTrn = cell(1,nClasses);
    inVal = cell(1,nClasses);
    for i=1:nClasses,
      inTrn{i} = in_trn{i} - ((W*in_trn{i})'*W)';
      inVal{i} = in_val{i} - ((W*in_val{i})'*W)';
    end
  else
    inTrn = in_trn;
    inVal = in_val;
  end
  
  
function [oNet, inTrn, inVal, inTst, sW] = forceOrthogonalization(net, trn, val, tst, saveWeights)
  oNet = net;
  inTrn = trn;
  inVal = val;
  inTst = tst;

  %If we  have already extracted a PCD, we remove
  % the information of the last PCD from the init values of the
  % new PCD to be extracted, and also from the input data.
  if size(net.IW{1},1) > 1,
    Nd = size(trn{1},1);
    
    %Getting the last PCD extracted.
    W = net.IW{1}(end-1,:);
    
    %Removing the info related to the PCD already extracted.
    Nc = length(trn);
    inTrn = cell(1,Nc);
    inVal = cell(1,Nc);
    inTst = cell(1,Nc);
    for i=1:Nc,
      inTrn{i} = trn{i} - ( repmat(W*trn{i},Nd,1) .* trn{i} );
      inVal{i} = val{i} - ( repmat(W*val{i},Nd,1) .* val{i} );
      inTst{i} = tst{i} - ( repmat(W*tst{i},Nd,1) .* tst{i} );
    end
  
    %Pointing the initial weights of the new PCD to the right direction.
    sW = saveWeights - ( (W*saveWeights') * saveWeights );
    oNet.IW{1}(end,:) = sW;
  else
    sW = net.IW{1};
  end


function [trnAlgo, maxNumPCD, numNodes, trfFunc, usingBias, trnParam] = getNetworkInfo(net)
  %Getting the network information regarding its topology

  %Taking the training algo.
  trnAlgo = net.trainFcn;

  %The maximum number of PCDs to be extracted is equal to the input size.
  maxNumPCD = net.inputs{1}.size;

  %Taking the other layer's size and training function.
  numNodes = [net.inputs{1}.size zeros(1,length(net.layers))];
  trfFunc = cell(1,length(net.layers));
  usingBias = zeros(1,length(net.layers));
  for i=1:length(net.layers),
    numNodes(i+1) = net.layers{i}.size;
    trfFunc{i} = net.layers{i}.transferFcn;
    usingBias(i) = net.layers{i}.userdata.usingBias;
  end
  
  trnParam = net.trainParam;
