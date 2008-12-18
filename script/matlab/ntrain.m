%function [outNet, epoch, trnError, tstError] = ntrain(net, in_trn, out_trn, in_tst, out_tst, trnESize)
%Trains a neural network using the network structure created by "newff2".
%Parameters are:
%	net             -> The neural network structure created by "newff2".
%	in_trn          -> The input training data.
%	out_trn         -> The output training data.
%	in_tst          -> The input testing data.
%	out_tst         -> The output testing data.
%	trnESize (opt)  -> The number of training events in an epoch.
%
%The data presented to the network must be organized so that each column is an event (either input or output),
%and the number of rows specifies the number of events to be presented to the neural network. If the number of 
%training events is not specified, the function will use, in each epoch, all the events in the 
%training group. In all cases, the events are randomly chosen within the training event set.
%
%function [outNet, epoch, trnError, tstError] = ntrain(net, in_trn, [], in_tst, [], trnESize)
%In this case, the training process will be optimized for pattern recognition problem.
%Parameters are:
%	net             -> The neural network structure created by "newff2".
%	in_trn          -> A cell array containing the input training data of each pattern.
%	in_tst          -> A cell array containing the input testing data of each pattern.
%	trnESize (opt)  -> A vector containing the number of training events in an epoch, for each pattern.
%
%The desired outputs (target) are internally generated, so, there is no need to provide the training
%and testing targets which can save a lot of memory. The input training and testing vectors are cell arrays with the
%same size as the number of patterns to be discriminated. Each cell must contain an array with the input events of an
%specific pattern (where each event is a column). The target output is organized so that the first output node is
%active for the first cell in the cell array, the second node is active for the second cell in the cell array, and so
%on. The training events per epoch, if supplied, must be a vector (1xNpat), where Npat is the number of 
%patterns to be recognized. Each element in this vectors will contain the number of events to be presented to the 
%network in each epoch, for each pattern.
%
%In every case, the function returns:
%	outNet -> The network structure with the new weight values obtained after training.
%	epoch    -> The epoch values.
%	trnError -> The training errors obtained for each epoch.
%	tstError -> The testing errors obtained for each epoch.
%