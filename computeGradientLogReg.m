function [ gradient ] = computeGradientLogReg( y, tX, beta )
% computes the MSE gradient for the given data points

gradient = tX'*(sigmoid(tX*beta) - y)/length(y);

end

