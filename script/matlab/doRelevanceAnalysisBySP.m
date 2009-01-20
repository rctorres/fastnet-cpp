function relevance = doRelevanceAnalysisBySP(net, electrons, jets)
%function relevance = doRelevanceAnalysis(net, electrons, jets)
%performs the relevance analysis over the variables of a dataset. For each
%run, one of the inputs is replaced by its mean value, and applied to the
%network net, so it can be analysed how relevant the input is comparing 
%the deviation (SP) obtained from the normal output. The function returns
%the deviation obtained for each input, in the same order as the inputs.
%
  
  [spVec, cutVec, detVec, faVec] = genROC(nsim(net, electrons), nsim(net, jets), 1000);
  idealSP = max(spVec);
  
  relevance = zeros(1,size(electrons,1));

  for i=1:size(electrons,1),
     auxE = electrons;
     auxJ = jets;
     auxE(i,:) = mean(electrons(i,:));
     auxJ(i,:) = mean(jets(i,:));
     [spVec, cutVec, detVec, faVec] = genROC(nsim(net, auxE), nsim(net, auxJ), 1000);
     relevance(i) = idealSP - max(spVec);
  end
