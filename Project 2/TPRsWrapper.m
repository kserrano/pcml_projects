function [ averageTPR, TPRVals, FPRVals ] = TPRsWrapper( labels, tXBeta )
% wrapper for TPR used in KCV for logistic / penalized logistic regression

predictions = probEstimate(tXBeta)*2-1; % gives classes as 1's and -1's
[ averageTPR, TPRVals, FPRVals ] = TPRs(labels, predictions);

end

