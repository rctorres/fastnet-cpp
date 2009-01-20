function [spRetVec, netVec] = trainMany(net, eTrn, eVal, eTst, jTrn, jVal, jTst, numTrains)
%function [spRetVec, netVec] = trainMany(net, eTrn, eVal, eTst, jTrn, jVal, jTst, numTrains)
%Returns the maximum SP value obtained for each training, as well as the
%trained network for each iteration. The function receives a non trained 
%(but configured) neural network net, the electrons and jets training 
%testing and validating sets and the number of times the
%network should be trained. Each time, its initial wheights are
%regenerated, to test how a network behaves for numTrains trains.
%
inTrn = {eTrn, jTrn};
inVal = {eVal, jVal};

spRetVec = zeros(1, numTrains);
netVec = [];

for i=1:numTrains,
  net = scrambleWeights(net);
  outNet = ntrain(net, inTrn, [], inVal, []);
  [spVec, cutVec, detVec, faVec] = genROC(nsim(outNet, eTst), nsim(outNet, jTst), 5000);
  spRetVec(i) = max(spVec);
  netVec = [netVec outNet];
end
