function net = newff2(nodesDist, trfFunc, useSP, netTrain)
%function net = newff2(nodesDist, trfFunc, useSP, netTrain)
%Creates a neural network structure just like the newff function (see help).
%Parameters are:
%  nodesDist 		-> The number of nodes in each layer (including the input).
%  trfFunc   		-> A cell vector containing the transfer function in each layer (excluding the input).
%  useSP (opt)     -> If true, then SP will be used for net goal.
%                       Otherwise, MSE. If ommited, MSE goal will be used.
%  netTrain (opt)  -> The training algorithm to be used. If none is specified, 'trainrp' will be used.
%
%The only modification in the net structure created, when compared to the one crated by newff
%is that this function adds new information to be used with the MatFastNet toolbox.
%Although the MatFastNet is full compatible with the network structures created with newff,
%you should try use newff2 in order to get full advantage of the MatFastNet toolbox.
%

if nargin == 2,
	netTrain = 'trainrp';
    useSP = false;
elseif nargin == 3,
    netTrain = 'trainrp';
end

aux = [-ones(nodesDist(1),1) ones(nodesDist(1),1)];
auxOut = [-ones(nodesDist(end),1) ones(nodesDist(end),1)];

%Creating the default network.
net = newff(aux, auxOut, nodesDist(2:(end-1)), trfFunc, netTrain);

%if the training is 'traingd', we add the decFactor parameter.
if (strcmp(net.trainFcn, 'traingd') == 1),
	net.trainParam.decFactor = 1;
end

%Adding the layer start and end values for the input layer.
net.inputs{1}.userdata.initNode = 1;
net.inputs{1}.userdata.endNode = net.inputs{1}.size;

%Adding the usingBias, frozen nodes and layer start and end values.
for i=1:net.numLayers,
	net.layers{i}.userdata.initNode = 1;
	net.layers{i}.userdata.endNode = net.layers{i}.size;
	net.layers{i}.userdata.usingBias = true;
	net.layers{i}.userdata.frozenNodes = [];
end

%Specifying the SP goal.
net.userdata.useSP = useSP;

%initializing the weights in the standard way, since the Matlab way sucks.
wInit = -0.2;
wEnd = 0.2;

%Doing the input layer.
net.IW{1} = unifrnd(wInit, wEnd, size(net.IW{1}));
net.b{1} = unifrnd(wInit, wEnd, size(net.b{1}));

%Doing the other layers.
for i=2:net.numLayers,
	net.LW{i,(i-1)} = unifrnd(wInit, wEnd, size(net.LW{i,(i-1)}));
	net.b{i} = unifrnd(wInit, wEnd, size(net.b{i}));
end
