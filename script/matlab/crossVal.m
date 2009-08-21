function [meanSP, stdSP] = crossVal(net, data, nBlocks, nDeal, nTrains)
%function [meanSP, stdSP] = crossVal(net, data, nBlocks, nDeal, nTrains)
%Performs cross validation analysis on the dataset data.
%Inputs parameters are:
% - net: The configured network to be trained during the cross validation
%        tests,
% - data: a cell vector where each cell is a matrix containing the events
%         to be used (one event per collumn).
% - nBlocks: specifies in how many blocks the data will be divided into.
% - nDeal: specifies how many times the blocks will be ramdomly distributed
%          into training, validation and test sets.
% - nTrains: specifies, for a given deal, how many times the network will
%            be trained, to avoid local minima.
%
%The function returns the mean and std value of the maximum SP obtained in
%each deal.
%
%WARNING: THIS FUNCTION ONLY WORKS FOR THE 2 CLASSES CASE!!!
%

if nargin < 3, nBlocks = 12; end
if nargin < 4, nDeal = 10; end
if nargin < 5, nTrains = 5; end

data = create_blocks(data, nBlocks);
netVec = get_networks(net, nTrains);
sp = zeros(1,nDeal);

for d=1:nDeal,
  [trn val tst] = deal_sets(data);
  sp(d) = get_best_train(netVec, trn, val, tst);
end

%Returning the mean and std of the SP obtained.
meanSP = mean(sp);
stdSP = std(sp);


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


function maxSP = get_best_train(net, trn, val, tst)
%Trains the network net multiple times, and returns the best SP obtained.
%The number of trains to perform is get from the length of the cell vector
%net.
  nTrains = length(net);
  sp = zeros(1, nTrains);
  for i=1:nTrains,
    onet = ntrain(net{i}, trn, val);
    out = nsim(onet, tst);
    [spVec, cutVec, detVec, faVec] = genROC(out{1}, out{2});
    sp(i) = max(spVec);
  end
  maxSP = max(sp);
