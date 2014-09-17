function r = relevance(net, trn, val, cut)
%function r = relevance(net, trn, val, cut)
%Perform relevance analysis. net can be iether a projection vector/matrix,
%or a neural network structure. trn/val can be a matrix, in which case the
%relevance calculation will be done via MSE, or a cell vector. If a cell
%vector (where each cell is the data corresponding to one pattern) is
%provided, then the relevance analysis will be done by SP.
%cut (optional with default to zero) is the cut threshold between classes.
%It is used only for relevance by SP.
%The function returns the output goal (MSE or SP) deviation for each removed input.
%
%WARNING: THIS FUNCTION WORKS FOR 2 CLASSES CASE ONLY
%

  if nargin < 4, cut = 0; end

  if iscell(trn),
    disp('Doing relevance analysis by SP.');
    mdata = mean(cell2mat(trn), 2);
    r = do_by_sp(net, val, mdata, cut);
  else
    disp('Doing relevance analysis by MSE.');
    mdata = mean(trn, 2);
    r = do_by_mse(net, val, mdata);
  end


function r = do_by_mse(net, data, mdata)
  %Does relevance by MSE.
  nDim = size(data,1);
  r = zeros(1,nDim);
  out_ref = get_output(net, data);
  
  for i=1:nDim,
     aux = data;
     aux(i,:) = mdata(i);
     out = get_output(net, aux);
     r(i) = mean( (out_ref - out).^2 );
  end

  
function r = do_by_sp(net, data, mdata, cut)
  %Does relevance by SP.
  nDim = size(data{1},1);
  r = zeros(1,nDim);
  sp_ref = get_sp(get_output(net, data{1}), get_output(net, data{2}), cut);
  
  for i=1:nDim,
     aux = data;
     aux{1}(i,:) = mdata(i);
     aux{2}(i,:) = mdata(i);
     sp = get_sp(get_output(net, aux{1}), get_output(net, aux{2}), cut);
     r(i) = sp_ref - sp;
  end


function out = get_output(net, data)
  %If net is a numeric matrix (a numeric fisher discriminat, for
  %instance, we simply perform the projection, otherwise, we call nsim,
  %since it is a neural network structure.
  %
  if isnumeric(net),
    out = net*data;
  else
    out = nsim(net, data);
  end

  
  function sp = get_sp(signal, noise, cut)
    [det, fa] = getEff(signal, noise, cut);
    sp = calcSP([det, (1-fa)]);
    
    