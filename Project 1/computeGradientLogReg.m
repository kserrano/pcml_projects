function [ gradient ] = computeGradientLogReg( y, tX, beta )
% computes the gradient for logistic regression, using the given formulas
% in the lectures

gradient = tX'*(sigmoid(tX*beta) - y)/length(y);

end

