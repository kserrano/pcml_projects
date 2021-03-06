function [pHat] = probEstimate(tXBeta)
% probEstimate compute the estimated probability from our inputs and beta
% (JHWT book page 135, section 4.4). The argument should be given as the
% product of tx*beta, directly, for generecity reasons

e = exp(tXBeta);
pHat = e./(1+e);

