function oNet = scrambleWeights(net)
%function oNet = scrambleWeights(net)
%Takes a neural network structure and scrmable it weights and biases
%matrices, returning the same network structure with the scrambled values.
%This function respects the frozen node information, which means that, if a
%node is frozen, the weights of the inputs and bias connected to it will
%NOT be scrambled.
%

%Scrambling the weights.
oNet = init(net);

%We now must recover the weights and bias values of the frozen nodes.

%Doing the input layer.
nodesIdx = net.layers{1}.userdata.frozenNodes;
oNet.IW{1}(nodesIdx,:) = net.IW{1}(nodesIdx,:);
oNet.b{1}(nodesIdx) = net.b{1}(nodesIdx);

%Doing the other layers.
for i=2:oNet.numLayers,
  nodesIdx = net.layers{i}.userdata.frozenNodes; 
	oNet.LW{i,(i-1)}(nodesIdx,:) = net.LW{i,(i-1)}(nodesIdx,:);
	oNet.b{i}(nodesIdx) = net.b{i}(nodesIdx);
end
