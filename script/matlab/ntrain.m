function [outNet, trnInfo] = ntrain(net, in_trn, out_trn, in_val, out_val, in_tst)
%function [outNet, trnInfo] = ntrain(net, in_trn, out_trn, in_val, out_val)
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
%function [outNet, trnInfo] = ntrain(net, in_trn, in_val, in_tst)
%In this case, the training process will be optimized for pattern recognition problem.
%Parameters are:
%	net             -> The neural network structure created by "newff2".
%	in_trn          -> A cell array containing the input training data of each pattern.
%	in_val          -> A cell array containing the input validating data of each pattern.
% in_tst          -> (OPTIONAL) The input testing data (Pattern Recognition ONLY).
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
%	trnInfo    -> A structure containing the training evolution information.
%

%Case pat rec net.
if (nargin == 3),
  %In this case, out_trn is, actually, the in_val.
  validateData(net, in_trn, out_trn);
  [outNet, trnInfo] = train_c(net, in_trn, [], out_trn, [], []);
elseif (nargin == 4),
  %In this case, out_trn is, actually, the in_val, and in_val is actually in_tst.
  validateData(net, in_trn, out_trn, in_val);
  [outNet, trnInfo] = train_c(net, in_trn, [], out_trn, [], in_val);
elseif (nargin == 5),
  validateData(net, in_trn, out_trn, in_val, out_val);
  [outNet, trnInfo] = train_c(net, in_trn, out_trn, in_val, out_val);
else
  error('Incorrect number of arguments! See help for information!');
end


function validateData(net, in_trn, out_trn, in_val, out_val)
  inputDim = net.inputs{1}.size;
  
  if (nargin == 3) || (nargin == 4),
    nClasses = length(in_trn);
    for i=1:nClasses,
      validateField(in_trn{i}, inputDim, sprintf('in_trn{%d}',i));
      validateField(out_trn{i}, inputDim, sprintf('in_val{%d}',i));
      if nargin == 4,
        validateField(in_val{i}, inputDim, sprintf('in_tst{%d}',i));
      end
    end
  elseif nargin == 5,
    validateField(in_trn, inputDim, 'in_trn');
    validateField(out_trn, inputDim, 'out_trn');
    validateField(in_val, inputDim, 'in_val');
    validateField(out_val, inputDim, 'out_val');
  end

function validateField(field, inputDim, id)
  if ~isa(field, 'double'),
    error(sprintf('%s is not a double precision matrix! Data must be of type "double"!', id));
  end
  
  if size(field,1) ~= inputDim,
    error(sprintf('%s does not have the same dimension as the network input layer!', id));
  end
 
