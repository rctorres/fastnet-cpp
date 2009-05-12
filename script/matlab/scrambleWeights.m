function outNet = scrambleWeights(net, forPCD)
%function scrambleWeights(net, forPCD)
%Takes a neural network structure and scrmable it weights and biases
%matrices, returning the same network structure with the scrambled values.
%This function respects the frozen node information, which means that, if a
%node is frozen, the weights of the inputs and bias connected to it will
%NOT be scrambled. if forPCD is true, then, the nodes of the input layer
%will NOT be scrambled. If ommited, forPCD = false.
%

if nargin == 1,
  forPCD = false;
end

outNet = net;

%initializing the weights in the standard way, since the Matlab way sucks.
wInit = -0.2;
wEnd = 0.2;

%Doing the input layer.
if ~forPCD,
  %This will make nodesIdx contain only the index of the nodes which are NOT frozen.
  [nextLayer, currLayer] = size(outNet.IW{1});
  nodesIdx = (1:nextLayer);
  nodesIdx(net.layers{1}.userdata.frozenNodes) = []; 
  nNodes = length(nodesIdx);

  outNet.IW{1}(nodesIdx,:) = unifrnd(wInit, wEnd, nNodes, currLayer);
  outNet.b{1}(nodesIdx) = unifrnd(wInit, wEnd, nNodes, 1);
end

%Doing the other layers.
for i=2:outNet.numLayers,
  [nextLayer, currLayer] = size(outNet.LW{i,(i-1)});
  nodesIdx = (1:nextLayer);
  nodesIdx(net.layers{i}.userdata.frozenNodes) = []; 
  nNodes = length(nodesIdx);

	outNet.LW{i,(i-1)}(nodesIdx,:) = unifrnd(wInit, wEnd, nNodes, currLayer);
	outNet.b{i}(nodesIdx) = unifrnd(wInit, wEnd, nNodes, 1);
end
