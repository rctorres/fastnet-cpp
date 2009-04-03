function [outNet, epoch, trnError, valError] = npcd(net, in_trn, out_trn, in_val, out_val, numTreads)
%HELP DA NPCD

%Getting the network information regarding its topology
numNodes = [net.inputs{1}.size];
trnFunc = ;
for i=1:length(net.layers),
  numNodes = [numNodes net.layers{i}.size];
  trnFunc = [trnFunc {net.layers{i}.transferFcn}]
end

