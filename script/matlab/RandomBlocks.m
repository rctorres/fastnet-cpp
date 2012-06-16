classdef RandomBlocks
%This class implements the sorted blocks algorithm for cross validation.
%It divides the input data in a set of blocks, and, at eachd eal, ramdomly
%selects a number of blocks for the training, validation and testing sets.
  properties (SetAccess = private)
    nTrn;
    nVal;
    nTst;
    blocks;
    tstIsVal;
  end
  
  methods
    function self = RandomBlocks(data, nTrn, nVal, nTst)
    %function self = RandomBlocks(data, nTrn, nVal, nTst)
    %Class constructor. Receives:
    %  - data. A cell vector, where each cell holds de events of a given class.
    %  - nTrn: the number of blocks (per class) selected for composing the training set
    %  - nVal: the number of blocks (per class) selected for composing the validation set
    %  - nTst: the number of blocks (per class) selected for composing the
    %          testing set. if nTst = 0, the clas wil enforce the testing
    %          set to be the same as the validation set.
    
      self.nTrn = nTrn;
      self.nVal = nVal;
      self.nTst = nTst;
      nBlocks = self.nTrn + self.nVal + self.nTst;
      self.blocks = self.create_blocks(data, nBlocks);

      %Taking the total number of blocks.
      fprintf('Numbers of blocks for cross validation: %d\n', nBlocks);
      fprintf('    Training blocks   : %d\n', self.nTrn);
      fprintf('    Validation blocks : %d\n', self.nVal);
      fprintf('    Testing blocks    : %d\n', self.nTst);
      
      self.tstIsVal = (self.nTst == 0);
      if self.tstIsVal,
        fprintf('    Enforcing tst = val!\n');
      end
    end
        
    
    function [trn val tst] = deal_sets(self)
    %function [trn val tst] = deal_sets(self)
    % Ramdomly selects the blocks for composing the training, validation
    % and testing sets. If the number of testing blocks is zero, the method
    % will make tst = val. The class return:
    % - trn : a cell vector where each cell hold the training set for each class.
    % - val : a cell vector where each cell hold the validation set for each class.
    % - tst : a cell vector where each cell hold the testing set for each class.

      [nClasses, nBlocks] = size(self.blocks);
      trn = cell(1,nClasses);
      val = cell(1,nClasses);
      tst = cell(1,nClasses);
    
      for c=1:nClasses,
        %Ramdonly sorting the blocks order.
        idx = randperm(nBlocks);

        ipos = 1;
        epos = self.nTrn;
        trn{c} = cell2mat(self.blocks(c,idx(ipos:epos)));
    
        ipos = epos + 1;
        epos = ipos + self.nVal - 1;
        val{c} = cell2mat(self.blocks(c,idx(ipos:epos)));
    
        if self.nTst ~= 0,
          ipos = epos + 1;
          epos = ipos + self.nTst - 1;
          tst{c} = cell2mat(self.blocks(c,idx(ipos:epos)));
        else
          tst{c} = val{c};
        end
      end
    end
  end

  methods (SetAccess = private)
    function bdata = create_blocks(self, data, nBlocks)
    %function bdata = create_blocks(self, data, nBlocks)
    %Split the data into nBlocks per class and return the data separated in
    %blocks.
      nClasses = length(data);
      bdata = cell(nClasses, nBlocks);
  
      %Ramdomly placing the events within the blocks.
      for c=1:nClasses,
        for b=1:nBlocks,
          bdata{c,b} = data{c}(:,b:nBlocks:end);
        end
      end
    end
  end
  
end