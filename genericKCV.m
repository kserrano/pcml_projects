function [errorTr, errorTe] = genericKCV( y, tX,...
    fittingFunction, errorFunction, lambdas, K, alpha, noOfSeeds)
% generic cross-validation function used to estimate the best parameter
% (the one giving the best Test/Train error pair. Can only test 2
% parameters at simultaneously at most, do NOT use vary more.
% If no parameters to validate is given, then the function simply gives back
% a cross-validation issued estimate of the test Error

% split data in K fold (we will only create indices)

% if lambdas is empty, then simply the code validates other parameters or
% estimates test error
noOfLambdas = max(1, length(lambdas));

errorTr = zeros(noOfSeeds, K, noOfLambdas);
errorTe = zeros(noOfSeeds, K, noOfLambdas);

% K-fold cross validation
for s = 1:noOfSeeds
    
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
            errorTr(s, k, i) = errorFunction(yTr, tXTr*beta);

            % testing MSE using least squares
            errorTe(s, k, i) = errorFunction(yTe, tXTe*beta);

        end
    end
end

errorTr = mean(errorTr, 2);
errorTe = mean(errorTe, 2);

errorTr = permute(errorTr, [1 3 2]);
errorTe = permute(errorTe, [1 3 2]);

end

