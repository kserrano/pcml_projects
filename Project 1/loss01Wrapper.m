function [ error01 ] = loss01Wrapper(y, txBeta)
% wrapper for 01 loss to have the same header as other error functions

p = probEstimate(txBeta);
yhat = p > 0.5;
error01 = loss01( y, yhat );

end

