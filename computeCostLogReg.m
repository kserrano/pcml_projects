function [ L ] = computeCostLogReg( y, tX, beta )
% computes the MLE cost function for the given data points

L = -(y'*tX*beta + sum(-log(1+exp(tX*beta))))/length(y);

end

