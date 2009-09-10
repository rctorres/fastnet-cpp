function ret = crossVal(data, net, nBlocks, nDeal, nTrains)
%function ret = crossVal(data, net, nBlocks, nDeal, nTrains)
%Performs cross validation analysis on the dataset data.
%Inputs parameters are:
% - data: a cell vector where each cell is a matrix containing the events
%         to be used (one event per collumn).
% - net: The configured network to be trained during the cross validation
%        tests. If ommited or [], a cross validation using a Fisher
%        classifier will be used. The fisher approach is ONLY valid if you
%        have only 2 classes!
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
%
%WARNING: THIS FUNCTION ONLY WORKS FOR THE 2 CLASSES CASE!!!
%

if nargin < 2, net = []; end
if nargin < 3, nBlocks = 12; end
if nargin < 4, nDeal = 10; end
if nargin < 5, nTrains = 5; end

data = create_blocks(data, nBlocks);

nROC = 500;
ret.net = cell(1,nDeal);
ret.sp = zeros(1,nDeal);
ret.det = zeros(nDeal, nROC);
ret.fa = zeros(nDeal, nROC);

if isempty(net),
  for d=1:nDeal,
    [trn val tst] = deal_sets(data);
    [ret.net{d} ret.sp(d) ret.det(d,:) ret.fa(d,:)] = get_sp_by_fisher(trn, tst, nROC);
  end  
else
  netVec = get_networks(net, nTrains);
  for d=1:nDeal,
    [trn val tst] = deal_sets(data);
    [ret.net{d} ret.sp(d) ret.det(d,:) ret.fa(d,:)] = get_best_train(netVec, trn, val, tst, nROC);
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
  

function [trn val tst] = deal_sets(data)
%Create the training, validation and test sets based on how many blocks per
%set we should have.

  [nClasses, nBlocks] = size(data);
  trn = cell(1,nClasses);
  val = cell(1,nClasses);
  tst = cell(1,nClasses);
  
  for c=1:nClasses,
    idx = randperm(nBlocks);
    trn{c} = cell2mat(data(c,idx(1:3:end)));
    val{c} = cell2mat(data(c,idx(2:3:end)));
    tst{c} = cell2mat(data(c,idx(3:3:end)));
  end
  
  
function netVec = get_networks(net, numCopies)
%Get a passed neural network and generates nCopies of each. Each copy will
%have a different weights initialization.
  netVec = cell(1,numCopies);
  for i=1:numCopies,
    netVec{i} = scrambleWeights(net);
  end


function [onet osp odet ofa] = get_best_train(net, trn, val, tst, nROC)
%Trains the network net multiple times, and returns the best SP obtained.
%The number of trains to perform is get from the length of the cell vector
%net.
  nTrains = length(net);

  netVec = cell(1, nTrains);
  sp = zeros(1, nTrains);
  det = zeros(nTrains, nROC);
  fa = zeros(nTrains, nROC);
  
  for i=1:nTrains,
    netVec{i} = ntrain(net{i}, trn, val);
    out = nsim(netVec{i}, tst);
    [spVec, cutVec, det(i,:), fa(i,:)] = genROC(out{1}, out{2}, nROC);
    sp(i) = max(spVec);
  end
  
  [maxSP, idx] = max(sp);
  onet = netVec{idx};
  osp = sp(idx);
  odet = det(idx,:);
  ofa = fa(idx,:);

  
function [w maxSP det fa] = get_sp_by_fisher(trn, tst, nROC)
%Calculates the best SP achieved considering a Fisher discriminant.
  w = fisher(trn{1}, trn{2});
  out = {w*tst{1}, w*tst{2}};
  [spVec, cutVec, det, fa] = genROC(out{1}, out{2}, nROC);
  maxSP = max(spVec);
