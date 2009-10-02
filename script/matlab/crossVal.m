function ret = crossVal(data, net, pp, tstIsVal, nBlocks, nDeal, nTrains)
%function ret = crossVal(data, net, pp, tstIsVal, nBlocks, nDeal, nTrains)
%Performs cross validation analysis on the dataset data.
%Inputs parameters are:
% - data: a cell vector where each cell is a matrix containing the events
%         to be used (one event per collumn).
% - net: The configured network to be trained during the cross validation
%        tests. If ommited or [], a cross validation using a Fisher
%        classifier will be used. The fisher approach is ONLY valid if you
%        have only 2 classes!
% - pp :      A structure containing 2 fileds named 'func' and 'par'.
%            'func' must be a pointer to a pre-processing function to be 
%             executed on the data. 'par' must be a structure containing 
%             any parameter that must be used by func. The calling
%             procedure is [trn,val,tst,] = pp.func(trn, val, tst, pp.par)
%             if func does not use any par, pp.par must be [].
% - tstIsVal : If true, in each deal, the data will be split into trn and
%             val only, and tst = val. Default is FALSE. 
% - nBlocks: specifies in how many blocks the data will be divided into.
% - nDeal: specifies how many times the blocks will be ramdomly distributed
%          into training, validation and test sets.
% - nTrains: specifies, for a given deal, how many times the network will
%            be trained, to avoid local minima. If net = [], this parameter
%            is ignored.
%
%The function returns a structure containing the following fields:
% - net : the best discriminator obtained for each deal. If a numerica
%         Fisher is used, then net is the projection achieved in each deal.
% - sp  : The maximum SP value achieved in each deal.
% - det : The detection efficiency values for the ROC curve, for each deal.
% - fa : The values for the false alarm for the ROC curve, for each deal.
% - pp : Pre-processing structure returned by pp_func.
%
%WARNING: THIS FUNCTION ONLY WORKS FOR THE 2 CLASSES CASE!!!
%

if nargin < 2, net = []; end
if (nargin < 3) || (isempty(pp)),
  pp.func = @do_nothing;
  pp.par = [];
end
if nargin < 4, tstIsVal = false; end
if nargin < 5, nBlocks = 12; end
if nargin < 6, nDeal = 10; end
if nargin < 7, nTrains = 5; end
if nargin > 7, error('Invalid number of parameters. See help!'); end

data = create_blocks(data, nBlocks);

nROC = 500;
ret.net = cell(1,nDeal);
ret.pp = cell(1,nDeal);
ret.sp = zeros(1,nDeal);
ret.det = zeros(nDeal, nROC);
ret.fa = zeros(nDeal, nROC);

if isempty(net),
  for d=1:nDeal,
    [trn val tst] = deal_sets(data, tstIsVal);
    [trn val tst ret.pp{d}] = pp.func(trn, val, tst, pp.par);
    [ret.net{d} ret.sp(d) ret.det(d,:) ret.fa(d,:)] = get_sp_by_fisher(trn, tst, nROC);
  end  
else
  ret.evo = cell(1,nDeal);
  netVec = get_networks(net, nTrains);
  for d=1:nDeal,
    [trn val tst] = deal_sets(data, tstIsVal);
    [trn val tst ret.pp{d}] = pp.func(trn, val, tst, pp.par);
    [ret.net{d} ret.evo{d} ret.sp(d) ret.det(d,:) ret.fa(d,:)] = get_best_train(netVec, trn, val, tst, nROC);
  end
end


function bdata = create_blocks(data, nBlocks)
%Creating the blocks. bdata{c,b}, where c is the class idx, and b is the
%block idx.
%
  nClasses = length(data);
  bdata = cell(nClasses, nBlocks);
  
  %Ramdomly placing the events within the blocks.
  for c=1:nClasses,
    for b=1:nBlocks,
      bdata{c,b} = data{c}(:,b:nBlocks:end);
    end
  end
  

function [trn val tst] = deal_sets(data, tstIsVal)
%Create the training, validation and test sets based on how many blocks per
%set we should have.

  [nClasses, nBlocks] = size(data);
  trn = cell(1,nClasses);
  val = cell(1,nClasses);
  tst = cell(1,nClasses);
  
  for c=1:nClasses,
    idx = randperm(nBlocks);
    if tstIsVal,
      trn{c} = cell2mat(data(c,idx(1:2:end)));
      val{c} = cell2mat(data(c,idx(2:2:end)));
      tst{c} = val{c};
    else
      trn{c} = cell2mat(data(c,idx(1:3:end)));
      val{c} = cell2mat(data(c,idx(2:3:end)));
      tst{c} = cell2mat(data(c,idx(3:3:end)));
    end
  end
  
  
function netVec = get_networks(net, numCopies)
%Get a passed neural network and generates nCopies of each. Each copy will
%have a different weights initialization.
  netVec = cell(1,numCopies);
  for i=1:numCopies,
    netVec{i} = scrambleWeights(net);
  end


function [onet oevo osp odet ofa] = get_best_train(net, trn, val, tst, nROC)
%Trains the network net multiple times, and returns the best SP obtained.
%The number of trains to perform is get from the length of the cell vector
%net.
  nTrains = length(net);

  netVec = cell(1, nTrains);
  evo = cell(1, nTrains);
  sp = zeros(1, nTrains);
  det = zeros(nTrains, nROC);
  fa = zeros(nTrains, nROC);
  
  for i=1:nTrains,
    [netVec{i} evo{i}]  = ntrain(net{i}, trn, val);
    out = nsim(netVec{i}, tst);
    [spVec, cutVec, det(i,:), fa(i,:)] = genROC(out{1}, out{2}, nROC);
    sp(i) = max(spVec);
  end
  
  [maxSP, idx] = max(sp);
  onet = netVec{idx};
  oevo = evo{idx};
  osp = sp(idx);
  odet = det(idx,:);
  ofa = fa(idx,:);

  
function [w maxSP det fa] = get_sp_by_fisher(trn, tst, nROC)
%Calculates the best SP achieved considering a Fisher discriminant.
  w = fisher(trn{1}, trn{2});
  out = {w*tst{1}, w*tst{2}};
  [spVec, cutVec, det, fa] = genROC(out{1}, out{2}, nROC);
  maxSP = max(spVec);

  
function [otrn, oval, otst, pp] = do_nothing(trn, val, tst, par)
%Dummy function to work with pp_function ponter.
  disp('Applying NO pre-processing...');
  otrn = trn;
  oval = val;
  otst = tst;
  pp = [];
  