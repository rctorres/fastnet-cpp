function [oNet, I] = trainMany(net, inTrn, inVal, inTst, numTrains, batchSize)
%function [oNet, I] = trainMany(net, inTrn, inVal, inTst, numTrains, batchSize)
%Returns the maximum SP value obtained for each training, as well as the
%trained network for each iteration. The function receives a non trained 
%(but configured) neural network net, the training 
%testing and validating sets and the number of times the
%network should be trained. Each time, its initial wheights are
%regenerated, to test how a network behaves for numTrains trains. oNEt is a structured
%vector containning the max sp obtained for the network, the trained network structure,
%the epochs evolution, and the training and validation errors obtained for each epoch.
%"I" is the index within the oNet cell vector where the best train was achieved. Finally,
% batchSize is the number of events to use, for each epoch. If not specified, 10000 events will
%be used, per epoch.
%

if nargin == 5,
  batchSize = 10000;
end

oNet = cell(1,numTrains);
spVec = zeros(1, numTrains);

for i=1:numTrains,
  net = scrambleWeights(net);
  [aux.net, aux.epoch, aux.trnError, aux.valError] = ntrain(net, inTrn, inVal, batchSize);
  aux.sp = calcSP(diag(genConfMatrix(nsim(aux.net, inTst))));
  spVec(i) = aux.sp;
  oNet{i} = aux;
end

[v, I] = max(spVec);
