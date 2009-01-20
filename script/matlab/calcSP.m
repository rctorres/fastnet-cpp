function ret  = calcSP(detVec, faVec)
%function ret  = calcSP(detVec, faVec) - Calculates the normalized [0,1] SP value.
%detVec is the detection efficiency vector, and faVec is the false alarm probabilities.
%

ret = sqrt(detVec .* (1-faVec)) .* ( (detVec + (1-faVec)) ./ 2 );
