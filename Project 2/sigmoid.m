function [ Y ] = sigmoid( X )
% implements the sigmoid function used in logistic regression
Y = 1./(1+exp(-X));

end

