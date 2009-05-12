function [oNet, I] = trainMany(net, inTrn, inVal, inTst, numTrains, forPCD)
%function [oNet, I] = trainMany(net, inTrn, inVal, inTst, numTrains, forPCD)
%Returns the maximum SP value obtained for each training, as well as the
%trained network for each iteration. The function receives a non trained 
%(but configured) neural network net, the training 
%testing and validating sets and the number of times the
%network should be trained. Each time, its initial wheights are
%regenerated, to test how a network behaves for numTrains trains. oNEt is a structured
%vector containning the max sp obtained for the network, the trained network structure,
%the epochs evolution, and the training and validation errors obtained for each epoch.
%"I" is the index within the oNet cell vector where the best train was achieved.
%If forPCD = true, then at each iteration, the weights connected to the
%input of the last node of the first hidden layer will be kept, at each 
%initialization, just like they were passed by the input parameter net.
%

if nargin == 5,
  forPCD = false;
end

if forPCD,
  savedWeights = net.IW{1}(end,:);
end

oNet = cell(1,numTrains);
spVec = zeros(1, numTrains);

for i=1:numTrains,
  net = scrambleWeights(net, forPCD);
  
  %Restoring the passed init values for this node, if extracting PCD.
  if forPCD,
    net.IW{1}(end,:) = savedWeights;
  end
  
  [aux.net, aux.epoch, aux.trnError, aux.valError] = ntrain(net, inTrn, inVal);
  aux.sp = calcSP(diag(genConfMatrix(nsim(aux.net, inTst))));
  spVec(i) = aux.sp;
  oNet{i} = aux;
end

[v, I] = max(spVec);
