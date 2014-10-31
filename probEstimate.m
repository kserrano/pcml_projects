function [P] = probEstimate(tX, beta)
% probEstimate compute the estimated probability from our inputs and beta
% (JHWT book page 135, section 4.4)

e = exp(tX'*beta);
P = e./(1+e);

