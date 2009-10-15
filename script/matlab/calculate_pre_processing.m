function [trn, val, tst, pproc] = calculate_pre_processing(trn, val, tst, pp)
%function [trn, val, tst, pproc] = calculate_pre_processing(trn, val, tst, pp)
%Helper function to perform the chain of pre-processing. USed by crossVal.
%

  [trn, val, tst, pproc] = pp.func(trn, val, tst, pp.par);
  
  if isfield(pp, 'relevComp'),
    
    disp('Getting the most relevant components.');
    for i=1:length(trn),
      trn{i} = trn{i}(pp.relevComp,:);
      val{i} = val{i}(pp.relevComp,:);
      tst{i} = tst{i}(pp.relevComp,:);
    end
    
    %Adding this new pre-processing to the pre-proc chain.
    p.name = 'relevance';
    p.relevComp = pp.relevComp;
    if iscell(pproc),
      pproc = [pproc p];
    else
      pproc = {pproc p};
    end
  else
    disp('I will NOT perform any reduction by relevance.');
  end
