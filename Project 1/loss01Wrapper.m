function [ error01 ] = loss01Wrapper(y, txBeta)
% wrapper for 01 loss to have the same header as other error functions

yhat = txBeta > 0;
error01 = loss01( y, yhat );

end

