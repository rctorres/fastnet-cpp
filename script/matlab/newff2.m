function net = newff2(inTrn, outData, hNodes, trfFunc, netTrain)
%function net = newff2(inTrn, outData, hNodes, trfFunc, netTrain)
%Creates a neural network structure just like the newff function (see help).
%Parameters are:
%  inData          -> The input training set. Each collumn is an event.
%  outData         -> The desired output for each training event. Each collumn
%                     is an event.
%  hNodes          -> The vector with the number of nodes in the hidden layers.
%                     If zero, a simple input->output layer will be created.
%  trfFunc         -> A cell vector containing the transfer function in each layer (excluding the input).
%  netTrain (opt)  -> The training algorithm to be used. If none is specified, 'trainrp' will be used.
%
%The only modification in the net structure created, when compared to the one crated by newff
%is that this function adds new information to be used with the FastNet toolbox.
%Although a network created with newff2 will train correctly with matlab NN
%toolbox, the opposite is not true.
%

  if nargin < 2, outData = [-1 1]; end
  if nargin < 3, hNodes = []; end
  if nargin < 4, trfFunc = {'tansig'}; end
  if nargin < 5, netTrain = 'trainrp'; end

  %Creating the default network.
  net = newff(fmtData(inTrn), fmtData(outData), hNodes, trfFunc, netTrain, 'learngdm', 'mse', {}, {}, 'divideind');

  %if the training is 'traingd', we add the decFactor parameter.
  if (strcmp(net.trainFcn, 'traingd') == 1),
    net.trainParam.decFactor = 1;
  end

  %Adding the usingBias and frozen nodes.
  for i=1:net.numLayers,
    net.layers{i}.userdata.usingBias = true;
    net.layers{i}.userdata.frozenNodes = [];
  end

  %Specifying the SP goal.
  net.trainParam.useSP = false;

  %Specifying the batch size.
  net.trainParam.batchSize = 10;


function fmtData = fmtData(data)
  if iscell(data),
    fmtData = minmax(cell2mat(data));
  else
    fmtData = minmax(data);
  end
