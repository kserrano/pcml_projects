function [errorsTr, errorsTe] = genericKCV( y, inputX,...
    fittingFunction, errorFunction, parameter, K, alpha, noOfSeeds,...
    constantNNoptions, normalizeFlag)
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

% inputX variable can be either training data X_train with the column of
% ones as a prefix (for most fitting functions) or simply X_train (for NN
% for example)

% if lambdas is empty, then simply the code validates other parameters or
% estimates test error
noOfParamVals = max(1, length(parameter));

errorsTr = zeros(noOfSeeds, K, noOfParamVals);
errorsTe = zeros(noOfSeeds, K, noOfParamVals);

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
    
    for i = 1:noOfParamVals
        for k = 1:K
            % get k'th subgroup in test, others in train
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);
            yTe = y(idxTe);
            tXTe = inputX(idxTe,:);
            yTr = y(idxTr);
            tXTr = inputX(idxTr,:);
            
            fittingFunctionType = nargin(fittingFunction);
            
            % handling normalization
            % this ASSUMES that all input variables are gaussian
            % distributed
            switch fittingFunctionType
                case 9
                    % these cases correspond to the cases where inputX = tX
                    % ( has a column of ones in the beginning
                    
                    % start counting at second column the input variables, because first one is a dummy column of 1's
                    nonNormalInputVars = nonNormalInputVars + 1;
                    
                    [tXTr, mu, sigma] = zscore(tXTr); % train, get mu and std
                    tXTe = normalize(tXTe, mu, sigma);  % normalize test data
                   
                otherwise
                    % these cases correspond to the cases where inputX = tX
                    % ( has a column of ones in the beginning
                    
                    [normXTr, mu, sigma] = zscore(tXTr(:, 2:end)); % train, get mu and std
                    normXTe = normalize(tXTe(:, 2:end), mu, sigma);  % normalize test data
                    
                    tXTr = [ones(size(normXTr, 1), 1) normXTr];
                    tXTe = [ones(size(normXTe, 1), 1) normXTe];
            end

            switch fittingFunctionType
                case 2 % only direct LS uses 2 args
                    beta = fittingFunction(yTr, tXTr);                    
                    
                case 3 % LS using GD, log. reg. and ride reg. use 3 args
                    if ~isempty(alpha) % this means it's either LS GD or log. reg, same input args
                        beta = fittingFunction(yTr, tXTr, alpha);
                    else %otherwise it must be ridge regression
                        beta = fittingFunction(yTr, tXTr, parameter(i));
                    end
                case 4 % only penalized log. reg. used 4 arguments
                    beta = fittingFunction(yTr, tXTr, alpha, parameter(i));
                case 9 % only NN takes so many input arguments !
                    
                    dimensions = constantNNoptions.dimensions;
                    noEpochs = constantNNoptions.noEpochs;
                    batchSize = constantNNoptions.batchSize;
                    learningRate = constantNNoptions.learningRate;
                    
                    % the data size needs to be a multiple of the batchsize
                    numTrSampToUse = batchSize * floor( size(tXTr) / batchSize);
                    
                    %numSampToUse = batchSize;
                    tXTr = tXTr(1:numTrSampToUse,:);
                    yTr = yTr(1:numTrSampToUse);
                    
                    if isfield(constantNNoptions, 'dropout')
                        % we're varying the L2 weight and keeping the
                        % dropout constant in this case
                        
                        dropout = constantNNoptions.dropout;
                        
                        NN = fittingFunction( tXtr, yTr, dimensions, noEpochs,...
                            batchSize, plotFlag, learningRate, dropout, parameter(i) );
                        
                    elseif isfield(constantNNoptions, 'L2Weight')      
                        % we're varying the dropout and keeping the
                        % L2 weight constant in this case
                        
                        L2Weight = constantNNoptions.L2Weight;
                        
                        NN = fittingFunction( tXtr, yTr, dimensions, noEpochs,...
                            batchSize, plotFlag, learningRate, parameter(i), L2Weight );
                        
                    else
                        error('constant NN options are missing either dropout value or L2 weight penalty')
                    end
                                         
            end
            
            % handling predictions
            switch fittingFunctionType
                case 9

                    % training set prediction
                    trainPred = predictNNBinaryOutput( tXTr, NN );

                    % testing set prediction
                    testPRed = predictNNBinaryOutput( tXTe, NN );
                otherwise

                    % all cases other than NN model can predict in
                    % the following way
                    trainPred = tXTr*beta; % training set prediction
                    testPRed = tXTe*beta; % testing set prediction
            end

            % training and test cost
            errorsTr(s, k, i) = errorFunction(yTr, trainPred);

            % testing MSE using least squares
            errorsTe(s, k, i) = errorFunction(yTe, testPRed);

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

