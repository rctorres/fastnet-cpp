function [detEff, faEff] = getEff(outSignal, outNoise, cut)
%function [detEff, faEff] = getEff(outSignal, outNoise, cut)
%Returns the detection and false alarm probabilities for a given input
%vector of detector's output for signal events(outSignal) and for noise 
%events (outNoise), using a decision threshold 'cut'. The result in whithin
%[0,1].
%

detEff = length(find(outSignal >= cut)) / length(outSignal);
faEff = length(find(outNoise >= cut)) / length(outNoise);
