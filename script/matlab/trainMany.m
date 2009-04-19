function [spRetVec, netVec] = trainMany(net, inTrn, inVal, inTst, numTrains)
%function [spRetVec, netVec] = trainMany(net, inTrn, inVal, inTst, numTrains)
%Returns the maximum SP value obtained for each training, as well as the
%trained network for each iteration. The function receives a non trained 
%(but configured) neural network net, the training 
%testing and validating sets and the number of times the
%network should be trained. Each time, its initial wheights are
%regenerated, to test how a network behaves for numTrains trains.
%
spRetVec = zeros(1, numTrains);
netVec = cell(1,numTrains);

for i=1:numTrains,
  net = scrambleWeights(net);
  outNet = ntrain(net, inTrn, inVal);
  [spVec, cutVec, detVec, faVec] = genROC(nsim(outNet, inTst{1}), nsim(outNet, inTst{2}), 5000);
  spRetVec(i) = max(spVec);
  netVec{i} = outNet;
end
