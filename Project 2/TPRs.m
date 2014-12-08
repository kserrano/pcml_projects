function [ averageTPR, TPRVals, FPRVals ] = TPRs( labels, scores )
% wrapper functions used for KCV that produces TPR measurements for given
% scores

% ASSUMING labels are given in binary 1's and -1's
[ averageTPR, ~, TPRVals, FPRVals ] = fastROC(labels > 0, scores,0, []);

end

