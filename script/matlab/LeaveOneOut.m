classdef LeaveOneOut
%This class implements the leave-one-out algorithm for cross validation.
%It will have as many deals as number of events. At each deal, an event
%will be selected as tst data, while all the others will be used for
%trn/val (forcing trn = val). 
  properties (SetAccess = private)
    data; %Will hold the events as a single matrix.
    nClasses; %Tells how many classes we have.
    class; %Will tell the class of each event so we will know how to split them.
    evIdx; %Will tell the index of the event to be left out.
  end
  
  methods
    function self = LeaveOneOut(data)
    %function self = LeaveOneOut(data)
    %Class constructor. Receives:
    %  - data. A cell vector, where each cell holds de events of a given class.
          
    self.evIdx = 1;
    self.data = [];
    self.class = [];
    self.nClasses = length(data);
    for i=1:self.nClasses,
        self.data = [self.data data{i}];
        self.class = [self.class i*ones(1,size(data{i},2))];
    end
      
      fprintf('Total number of classes                    : %d\n', self.nClasses);      
      fprintf('Total number of events for cross validation: %d\n', length(self.class));
    end
    
    
    
    function [trn val tst] = deal_sets(self)
    %function [trn val tst] = deal_sets(self)
    % Will remove one of the events, assigning it to tst, and will set the
    % others to be trn and val (trn = val).
    % - trn : a cell vector where each cell hold the training set for each class.
    % - val : a cell vector where each cell hold the validation set for each class.
    % - tst : a cell vector where each cell hold the testing set for each class.

      auxData = self.data;
      auxClass = self.class;
      trn = cell(1,self.nClasses);
      tst = cell(1,self.nClasses);
      
      %We first select the single event to compose the tst set
      tst{auxClass(self.evIdx)} = auxData(:,self.evIdx); 

      %Then we remove it from the set before creating the trn set.
      auxData(:,self.evIdx) = [];
      auxClass(self.evIdx) = [];
      
      %Creating the trn set with the rest, assigning them to each class.
      for c=1:self.nClasses,
          trn{c} = auxData(:, auxClass == c);
      end
      
      %Forcing trn = val;
      val = trn;
      
      %Moving to the next event.
      self.evIdx = self.evIdx + 1;
    end
  end  
end