classdef KMeansBlocks
%This class implements the k-means blocks algorithm for cross validation.
%It divides the input data in a set of clusters, and, at each deal, ramdomly
%selects a number of blocks for the training, validation and testing sets.
  properties (SetAccess = private)
    percTrn;
    percVal;
    percTst;
    clusters;
  end
  
  methods
    function self = KMeansBlocks(data, nClusters, percTrn, percVal, percTst)
    %function self = KMeansBlocks(data, nClusters, percTrn, percVal, percTst)
    %Class constructor. Receives:
    %  - data. A cell vector, where each cell holds de events of a given class.
    %  - nCluster: the number of clusters to create.
    %  - percTrn: the percentage of events (per class) for composing the training set
    %  - percVal: the percentage of events (per class) for composing the validation set
    %  - percTst: the percentage of events (per class) for composing the testing set
    %             testing set. if nTst = 0, the clas wil enforce the testing
    %             set to be the same as the validation set.
    
      self.percTrn = percTrn;
      self.percVal = percVal;
      self.percTst = percTst;
      
      perc = self.percTrn + self.percVal + self.percTst;
      if perc > 1,
        error('Sum of percentages is greater than 100%%!');
      elseif perc < 1,
        warning('Sum of percentages is smaller than 100%%!');
      end
      
      self.clusters = self.create_clusters(data, nClusters);

      %Taking the total number of blocks.
      fprintf('Numbers of clusters for cross validation: %d\n', nClusters);
      fprintf('    Percentage of training events   : %2.2f\n', 100*self.percTrn);
      fprintf('    Percentage of validation events : %2.2f\n', 100*self.percVal);
      fprintf('    Percentage of testing events    : %2.2f\n', 100*self.percTst);
      
      if self.tstIsVal(),
        fprintf('    Enforcing tst = val!\n');
      end
    end
        
    
    function ret = tstIsVal(self)
    %function ret = tstIsVal(self)
    %Returns true if the class is enforcing tst = val.
      ret = (self.percTst == 0);
    end
    
    
    function [trn val tst] = deal_sets(self)
    %function [trn val tst] = deal_sets(self)
    % Selects the trn, val and tst events respecting the percentage set for
    % the class. If percentual of tst events is zero, the method
    % will make tst = val. The class return:
    % - trn : a cell vector where each cell hold the training set for each class.
    % - val : a cell vector where each cell hold the validation set for each class.
    % - tst : a cell vector where each cell hold the testing set for each class.

      [nClasses, nClusters] = size(self.clusters);
      trn = cell(1,nClasses);
      val = cell(1,nClasses);
      tst = cell(1,nClasses);
    
      for c=1:nClasses,
        data_trn = [];
        data_val = [];
        data_tst = [];
        
        for clus=1:nClusters,
          [clus_trn, clus_val, clus_tst] = dividerand(self.clusters{c, clus}, self.percTrn, self.percVal, self.percTst);
          data_trn = [data_trn clus_trn];
          data_val = [data_val clus_val];
          data_tst = [data_tst clus_tst];
        end
        
        trn{c} = data_trn;
        val{c} = data_val;
        tst{c} = data_tst;
            
        if self.tstIsVal(),
          tst{c} = val{c};
        end
      end
    end
  end

  
  methods
    function bdata = create_clusters(self, data, nClusters)
    %function bdata = create_clusters(self, data, nClusters)
    %Create nClusters for each class using k-means clustering algorithm.
    %Returns a cell vector where each row represents a class, and each
    %column represents a cluster.
    %
      nClasses = length(data);
      bdata = cell(nClasses, nClusters);
  
      %Generating the clusters for each class.
      for c=1:nClasses,
        classData = data{c};
        idx = kmeans(classData', nClusters);
        
        %Splitting among each cluster
        for clus=1:nClusters,
          bdata{c,clus} = classData(:, (idx == clus));
        end
      end
    end
  end
  
end