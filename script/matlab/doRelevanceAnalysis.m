function relevance = doRelevanceAnalysis(net, dataSet)
%function relevance = doRelevanceAnalysis(net, dataSet)
%performs the relevance analysis over the variables of a dataset. For each
%run, one of the inputs is replaced by its mean value, and applied to the
%network net, so it can be analysed how relevant the input is comparing 
%the deviation (MSE) obtained from the normal output. The function returns
%the deviation obtained for each input, in the same order as the inputs.
%
  outIdeal = nsim(net, dataSet);
  relevance = zeros(1,size(dataSet,1));

  for i=1:size(dataSet,1),
     auxData = dataSet;
     auxData(i,:) = mean(dataSet(i,:));
     outMean = nsim(net, auxData);
     relevance(i) = mean( (outIdeal - outMean).^2 );
  end
