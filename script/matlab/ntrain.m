function [outNet, epoch, trnError, valError] = ntrain(net, in_trn, out_trn, in_val, out_val)
%function [outNet, epoch, trnError, valError] = ntrain(net, in_trn, out_trn, in_val, out_val)
%Trains a neural network using the network structure created by "newff2".
%Parameters are:
%	net             -> The neural network structure created by "newff2".
%	in_trn          -> The input training data.
%	out_trn         -> The output training data.
%	in_val          -> The input validating data.
%	out_val         -> The output validating data.
%
%The data presented to the network must be organized so that each column is an event (either input or output),
%and the number of rows specifies the number of events to be presented to the neural network. 
%
%function [outNet, epoch, trnError, valError] = ntrain(net, in_trn, in_val)
%In this case, the training process will be optimized for pattern recognition problem.
%Parameters are:
%	net             -> The neural network structure created by "newff2".
%	in_trn          -> A cell array containing the input training data of each pattern.
%	in_val          -> A cell array containing the input validating data of each pattern.
%
%The desired outputs (target) are internally generated, so, there is no need to provide the training
%and validating targets which can save a lot of memory. The input training and validating vectors are cell arrays with the
%same size as the number of patterns to be discriminated. Each cell must contain an array with the input events of an
%specific pattern (where each event is a column). The target output is organized so that the first output node is
%active for the first cell in the cell array, the second node is active for the second cell in the cell array, and so
%on.
%
%In every case, the function returns:
%	outNet -> The network structure with the new weight values obtained after training.
%	epoch    -> The epoch values.
%	trnError -> The training errors obtained for each epoch.
%	valError -> The validating errors obtained for each epoch.
%

%Case pat rec net.
if (nargin == 3),
  %In this case, out_trn is, actually, the in_val.
  validateDataType(in_trn, out_trn);
  [outNet, epoch, trnError, valError] = train_c(net, in_trn, [], out_trn, [], in_val);
elseif (nargin == 5),
  validateDataType(in_trn, out_trn, in_val, out_val);
  [outNet, epoch, trnError, valError] = train_c(net, in_trn, out_trn, in_val, out_val, batchSize);
else
  error('Incorrect number of arguments! See help for information!');
end


function validateDataType(in_trn, out_trn, in_val, out_val)
  if nargin == 2,
    nClasses = length(in_trn);
    for i=1:nClasses,
      if ~isa(in_trn{i}, 'single'),
        error(sprintf('in_trn{%d} is not a single precision matrix! Data must be of type "single"!', i));
      end
      if ~isa(out_trn{i}, 'single'), %out_trn is out in_val.
        error(sprintf('in_val{%d} is not a single precision matrix! Data must be of type "single"!', i));
      end  
    end
  elseif nargin == 4,
    if ~isa(in_trn, 'single'),
      error('in_trn is not a single precision matrix! Data must be of type "single"!');
    end
    if ~isa(out_trn, 'single'),
      error('out_trn is not a single precision matrix! Data must be of type "single"!');
    end
    if ~isa(in_val, 'single'),
      error('in_val is not a single precision matrix! Data must be of type "single"!');
    end
    if ~isa(out_val, 'single'),
      error('out_val is not a single precision matrix! Data must be of type "single"!');
    end
end

