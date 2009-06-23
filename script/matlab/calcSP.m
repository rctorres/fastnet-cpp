function ret  = calcSP(effic)
%function ret  = calcSP(effic) - Calculates the normalized [0,1] SP value.
%effic is a vector containing the detection efficiency [0,1] of each
%discriminating pattern. It effic is a Matrix, the SP will be calculated 
%for each collumn.
%

ret = geomean(effic) .* mean(effic);
