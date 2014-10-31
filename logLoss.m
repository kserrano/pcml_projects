function [lo] = logLoss(y, p)
%LOGODDS compute the log odds for an output vector y and a vector
%probability p

% Error if y and p have not the same length
if (length(y) ~= length(p))
    error(' y and p have not the length')
end
    
% y must be a vector
y = y(:);

% p must be a vector
p = p(:);

% check for the length of y
N = length(y);

% compute the logLoss
lo = -(1/N).*sum((y.*log(p)+(1-p).*log(1-p)));


