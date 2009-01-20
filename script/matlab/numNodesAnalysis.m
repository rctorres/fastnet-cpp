function [spMean, spStd, netStr] = numNodesAnalysis(numNodesVec, numTrains, eTrn, eVal, eTst, jTrn, jVal, jTst)
%function [spMean, spStd, netStr] = numNodesAnalysis(numNodesVec, numTrains, eTrn, eVal, eTst, jTrn, jVal, jTst)
%This function will take the default training, validating and testing sets for electrons and jets
%and will perform the hidden nodes validation, which means, varying the number of nodes in the 
%hidden layer taking the values in the numNodesVec. If a network with only one hidden node is
%requested, then the training is done, for this particular case, with NO hidden layer. For
%better statistics, for each case, the network is trained numTrains times. The function
%returns the mean and std values of the SPs obtained for each configuration.
%Also, the function returs an structure vector containing the networks for
%each case.
%

epochs = 3000;
max_fail = 20;
show = 0;
trfFunc = 'tansig';

if length(find(numNodesVec <= 0)) > 0,
  error('numNodesVec must have only positive values\n');
end

start = 1;
nNodes = length(numNodesVec);
nInput = size(eTrn,1);
numNodesVec = sort(numNodesVec);
inTrn = {eTrn, jTrn};
inVal = {eVal, jVal};
spMean = zeros(1,nNodes);
spStd = zeros(1,nNodes);
netStr = [];

if numNodesVec(1) == 1,
  fprintf('Training linear network (%d x 1)\n', nInput);
  net = newff2([nInput, 1], {trfFunc}, false, 'trainrp');
  net.trainParam.epochs = epochs;
  net.trainParam.show = show;
  net.trainParam.max_fail = max_fail;
  [spRetVec, netVec] = trainMany(net, eTrn, eVal, eTst, jTrn, jVal, jTst, numTrains);
  spMean(1) = mean(spRetVec);
  spStd(1) = std(spRetVec);
  start = 2;
  
  aux.nets = netVec;
  aux.numNodes = 1;
  netStr = [netStr aux];
end

for i=start:nNodes,
  fprintf('Training non-linear network (%d x %d x 1)\n', nInput, numNodesVec(i));
  net = newff2([nInput, numNodesVec(i), 1], {trfFunc, trfFunc}, false, 'trainrp');
  net.trainParam.epochs = epochs;
  net.trainParam.show = show;
  net.trainParam.max_fail = max_fail;
  [spRetVec, netVec] = trainMany(net, eTrn, eVal, eTst, jTrn, jVal, jTst, numTrains);
  spMean(i) = mean(spRetVec);
  spStd(i) = std(spRetVec);
  
  aux.nets = netVec;
  aux.numNodes = i;
  netStr = [netStr aux];
end

