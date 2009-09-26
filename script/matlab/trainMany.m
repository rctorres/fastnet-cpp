function [oNet, I] = trainMany(net, inTrn, inVal, inTst, numTrains)
%function [oNet, I] = trainMany(net, inTrn, inVal, inTst, numTrains)
%Returns the maximum SP value obtained for each training, as well as the
%trained network for each iteration. The function receives a non trained 
%(but configured) neural network net, the training 
%testing and validating sets and the number of times the
%network should be trained. Each time, its initial wheights are
%regenerated, to test how a network behaves for numTrains trains. oNEt is a structured
%vector containning the max sp obtained for the network, the trained network structure,
%the epochs evolution, and the training and validation errors obtained for each epoch.
%"I" is the index within the oNet cell vector where the best train was achieved.
%

oNet = cell(1,numTrains);
spVec = zeros(1, numTrains);
nClasses = length(inTrn);

for i=1:numTrains,
  net = scrambleWeights(net);
  [aux.net, aux.trnEvo] = ntrain(net, inTrn, inVal);
  out = nsim(aux.net, inTst);
  
  if nClasses > 2,
    aux.sp = calcSP(diag(genConfMatrix(out)));
  else
    aux.sp = max(genROC(out{1}, out{2}));
  end

  spVec(i) = aux.sp;
  oNet{i} = aux;
end

[v, I] = max(spVec);
