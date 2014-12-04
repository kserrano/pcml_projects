function [errorsTr, errorsTe] = genericKCV( y, tX,...
    fittingFunction, errorFunction, lambdas, K, alpha, noOfSeeds)
% generic cross-validation function used to estimate the best parameter
% (the one giving the best Test/Train error pair (more emphasis on the
% latter). It can be used with any fitting method (e.g. least squares) and
% any error metric (e.g. rmse) to cross validate a parameter lambda (for ridgeregression for example)
% or to simply estimate testError (in that case only a specific value of
% lambda should be used, not multiple ones). To actually get the test error
% at the end you simply need to take the mean of errorsTe in that case.
% Alpha should also be given when the fitting function requires it, and the
% noOfSeeds determines the number of test curves being averaged (more makes
% the cross-validation more stable, but takes longer). K is the number of
% folds

% if lambdas is empty, then simply the code validates other parameters or
% estimates test error
noOfLambdas = max(1, length(lambdas));

errorsTr = zeros(noOfSeeds, K, noOfLambdas);
errorsTe = zeros(noOfSeeds, K, noOfLambdas);

parpool

% K-fold cross validation
for s = 1:noOfSeeds
    
    % split data in K fold (we will only create indices)
    setSeed(s);
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    
    for i = 1:noOfLambdas
        for k = 1:K
            % get k'th subgroup in test, others in train
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);
            yTe = y(idxTe);
            tXTe = tX(idxTe,:);
            yTr = y(idxTr);
            tXTr = tX(idxTr,:);

            fittingFunctionType = nargin(fittingFunction);

            switch fittingFunctionType
                case 2 % only direct LS uses 2 args
                    beta = fittingFunction(yTr, tXTr);
                case 3 % LS using GD, log. reg. and ride reg. use 3 args
                    if ~isempty(alpha) % this means it's either LS GD or log. reg, same input args
                        beta = fittingFunction(yTr, tXTr, alpha);
                    else %otherwise it must be ridge regression
                        beta = fittingFunction(yTr, tXTr, lambdas(i));
                    end
                case 4 % only penalized log. reg. used 4 arguments
                    beta = fittingFunction(yTr, tXTr, alpha, lambdas(i));
            end

            % training and test cost
            errorsTr(s, k, i) = errorFunction(yTr, tXTr*beta);

            % testing MSE using least squares
            errorsTe(s, k, i) = errorFunction(yTe, tXTe*beta);

        end
    end
end

% averages over the k folds
errorsTr = mean(errorsTr, 2);
errorsTe = mean(errorsTe, 2);

% each line the output corresponds to a seed, and each column to a
% parameter (e.g. lambda) value
errorsTr = permute(errorsTr, [1 3 2]);
errorsTe = permute(errorsTe, [1 3 2]);

end

