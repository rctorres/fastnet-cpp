function [detEff, faEff] = getEff(outSignal, outNoise, cut)
%function ret = genConfMatrix(out, target)
%Returns the confusion matrix of the results obtained. The 'out' and
%'target' vectors are, respectively, the network's output and the desired
%(target) values, which one event per column. The resulting matrix places
%each class in a row, so if ret(2,3) = 5 means that the five events from class 2 
%where classified as being from class 3.
%

detEff = length(find(outSignal >= cut)) / length(outSignal);
faEff = length(find(outNoise >= cut)) / length(outNoise);
