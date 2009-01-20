function outNet = scrambleWeights(net)
%function scrambleWeights(net)
%Takes a neural network structure and scrmable it weights and biases
%matrices, returning the same network structure with the scrambled values.
%

outNet = net;

%initializing the weights in the standard way, since the Matlab way sucks.
wInit = -0.2;
wEnd = 0.2;

%Doing the input layer.
outNet.IW{1} = unifrnd(wInit, wEnd, size(outNet.IW{1}));
outNet.b{1} = unifrnd(wInit, wEnd, size(outNet.b{1}));

%Doing the other layers.
for i=2:outNet.numLayers,
	outNet.LW{i,(i-1)} = unifrnd(wInit, wEnd, size(outNet.LW{i,(i-1)}));
	outNet.b{i} = unifrnd(wInit, wEnd, size(outNet.b{i}));
end
