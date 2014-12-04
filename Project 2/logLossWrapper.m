function  errorLogLoss  = logLossWrapper( y, txBeta )
% wrapper for 01 loss to have the same header as other error functions

p = probEstimate(txBeta);
errorLogLoss = logLoss(y, p);

end

