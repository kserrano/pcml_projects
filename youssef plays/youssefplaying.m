% Comments and observations made by youssef

clc
clear all
close all

load('MexicoCity_regression.mat')
%load('MexicoCity_classification.mat')

N = length(y_train); % data size
D = size(X_train, 2); % dimensionality

%% plotting the histograms of the variables
noOfVertPlots = floor(sqrt(D));
noOfHorFigures = ceil(D/noOfVertPlots);

figure;
suptitle('histogram of all the input variables')
for i = 1:D
    subplot(noOfVertPlots, noOfHorFigures, i)
    hist(X_train(:, i))
    title(num2str(i))
end

figure;
hist(y_train);
title('hisogram of output variable')

%% categorical variable detection

noOfVariableValues = zeros(D, 1);

% SUPPOSING categorial variables have AT MOST 15 classes
maxClasses = 15;

categorialVariableClasses = cell(81, D+1);

for i=1:D
    temp = unique(X_train(:, i));
    categorialVariableClasses{i} = temp;
    noOfVariableValues(i) = length(temp);
end

catVars = find(noOfVariableValues < maxClasses);
categorialVariableClasses = categorialVariableClasses(catVars);

%% normalizing the features

% for regression variables
% normalize features (store the mean and variance)
meanX = mean(X_train);

% 42 first features seem to be gaussian distributed, can be normalized
normInputVars = setdiff(1:D, catVars);

% normalizing the input variables
X_train(:, normInputVars) = X_train(:, normInputVars) - repmat(meanX(normInputVars), N, 1);
stdX = std(X_train);
X_train(:, normInputVars) = X_train(:, normInputVars)./repmat(stdX(:, normInputVars), N, 1);

% normalizing the output variable, comment out for classification
y_train = (y_train-mean(y_train))/std(y_train); % for regression
%y_train = -y_train;
%y_train(y_train == -1) = 0; % for classification 

% Form (y,tX) to get regression data in matrix form
tX = [ones(N,1) X_train(:, :)];

%% replacing categorical variables with dummy ones

dummyCategoricalVariables = dummyvar(X_train(:, catVars)+1);
%txExt = [txExt(:, setdiff(1:(D+1), catVars)) dummyCategoricalVariables];

%% correlation & normal distribution analysis

% taking the 18th and 34th variables, and any polynomial transform of them,
% and checking their pearson correlation with the output

% use the folowing line for testing dummy variables's correlation with the
% output, comment out the next assignment to txSpecial
%txSpecial = dummyCategoricalVariables;

%txSpecial = [myPoly(X_train(:, 18),5) myPoly(X_train(:, 34),10) ];
%statistics = [y_train txSpecial];

% for testing all polynomials of the interesting 11th and 24th
% classification variables
%statistics = [y_train  myPoly(X_train(:, 11), 5) myPoly(X_train(:, 24), 5)];
statistics = [y_train  (X_train(:, 11)-min(X_train(:, 11))).^0.5 (X_train(:, 24)-min(X_train(:, 24))).^0.5];

% use this line if u want to plot the correlations for all the data instead
%statistics = [y_train X_train];

[ spearmanCoeffs, spearmanSig, pearsonCoeffs, pearsonSig,...
    normalityTests, corrExistence, corrSuitability] = ...
    computeCorrelationCoeffs( statistics, size(statistics, 2));

normalityTests = circshift(normalityTests, -1);

% figure;
% spearmanGraph = abs(spearmanSig.*spearmanCoeffs);
% stem(spearmanGraph)
% title('correlated variables with coeff, 0 otherwise, spearman, , in absolute value')

figure;
pearsonGraph = abs(pearsonSig.*pearsonCoeffs);
stem(pearsonGraph)
title('correlated variables with coeff, 0 otherwise, pearson, in absolute value')

% figure;
% stem(spearmanGraph.*pearsonGraph)
% title('correlated variables with coeff, 0 otherwise, pearson x spearman, in absolute value')
% 
% figure;
% stem(mean([spearmanGraph; pearsonGraph], 1))
% title('correlated variables with coeff, 0 otherwise, mean(pearson, spearman), in absolute value')

%% preliminary fitting (regression)

ranktX = rank(tX);
disp(' ');
disp(['rank of data columns is :' num2str(ranktX)]);

% toggling categorical variable encoding
%tX = tX(:,[1 (normInputVars+1)]);

% stupid mean fitting
betaMean = mean(y_train);
rmseWRTmean = rmse(y_train, betaMean);
disp(' ');
disp(['simply taking the mean gives Training rmse = ' num2str(rmseWRTmean)]);

% simple least squares
betaLS = leastSquares(y_train, tX);
y_hatLS = tX*betaLS;
LSrmse = rmse(y_hatLS, y_train);
disp(' ');
disp(['simple LS gives Training rmse = ' num2str(LSrmse)]);

alpha = 0.01;
betaLSGD = leastSquaresGD(y_train, tX, alpha);
y_hatLSGD = tX*betaLSGD;
LSGDrmse = rmse(y_hatLSGD, y_train);
disp(' ');
disp(['gradient LS gives Training rmse = ' num2str(LSGDrmse)]);

noOfLambdas = 10;
RRrmses = zeros(1, noOfLambdas);
lambdas = logspace(-2, 2, noOfLambdas);

disp(' ');
for i = 1:noOfLambdas

    betaRR = ridgeRegression(y_train, tX, lambdas(i));
    y_hatRR = tX*betaRR;
    RRrmses(i) = rmse(y_hatRR, y_train);
    disp(['for lambda = ' num2str(lambdas(i)) ' ridge regression fits with RMSE ' num2str(RRrmses(i))])
    
end

% usung the 18th and 34th variables transforms BE CAREFUL NOT TO REPRODUCE
% SAME COLUMN TWICE AS MYPOLY REPRODUCES A POLYNOMIAL OF DEGREE ONE (SO THE
% VARIABLE ITSELF)
txExt =[tX(:, setdiff(1:49, 35)) X_train(:, 18).^3 myPoly(X_train(:, 34), 6)];
[rmseTr, rmseTe] = genericKCV( y_train, txExt,...
@leastSquares, @rmse, [], 4, [], 100);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using transforms on 18th and 34th, gives test RMSE: ' num2str(meanErrTe)])

% txExt =[tX(:, setdiff(1:49, 35)) X_train(:, 18).^3 myPoly(X_train(:, 34), 6) (X_train(:, 34)-min(X_train(:, 34))).^0.5];
% [rmseTr, rmseTe] = genericKCV( y_train, txExt,...
% @leastSquares, @rmse, [], 4, [], 100);
% meanErrTe = mean(rmseTe);
% disp(' ')
% disp(['estimated test error using transforms on 18th and 34th + sqrt of 34th, gives test RMSE: ' num2str(meanErrTe)])

%% adding good dummy variables
clc

significantRegressionDummyVars = [29 39 41];

% removing categorical variables
txExt1 = [txExt(:, setdiff(1:size(txExt, 2), catVars))];
[rmseTr, rmseTe] = genericKCV( y_train, txExt1,...
@leastSquares, @rmse, [], 4, [], 100);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using transforms on 18th and 34th without categorical ones, gives test RMSE: ' num2str(meanErrTe)]);

% removing categorical variables and keeping only the 44th and 47th as they
% are
txExt2 = [txExt(:, setdiff(1:size(txExt, 2), catVars))  tX(:, [44 47]) ];
[rmseTr, rmseTe] = genericKCV( y_train, txExt2,...
@leastSquares, @rmse, [], 4, [], 100);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using transforms on 18th and 34th with only correlated categorical ones (taken as-is), gives test RMSE: ' num2str(meanErrTe)]);

% removing categorical variables and keeping only the 44th and 47th by
% takin all their issues dummy variables
txExt2 = [txExt(:, setdiff(1:size(txExt, 2), catVars))  dummyvar(tX(:, [44 47])+1) ];
[rmseTr, rmseTe] = genericKCV( y_train, txExt2,...
@leastSquares, @rmse, [], 4, [], 100);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using transforms on 18th and 34th with only correlated categorical ones (taking all their dummy variables), gives test RMSE: ' num2str(meanErrTe)]);

% removing categorical variables and keeping only the 44th and 47th by
% takin only their dummy variables that are correlated with the output
txExt3 = [txExt(:, setdiff(1:size(txExt, 2), catVars)) dummyCategoricalVariables(:, significantRegressionDummyVars) ];
[rmseTr, rmseTe] = genericKCV( y_train, txExt3,...
@leastSquares, @rmse, [], 4, [], 100);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using transforms on 18th and 34th with only correlated categorical ones (taking good classes only), gives test RMSE: ' num2str(meanErrTe)]);

%% removing kevin's bad (spurious, to ignore) input variables (regression)

regressionSpuriousVariables = [27    30    31    33    39    44    47];

txExt3 = [tX(:, setdiff(1:size(tX, 2), regressionSpuriousVariables+1))];
[rmseTr, rmseTe] = genericKCV( y_train, txExt3,...
@leastSquares, @rmse, [], 4, [], 100);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error when removing spurious variables, gives test RMSE: ' num2str(meanErrTe)]);

txExt4 = [tX(:, setdiff(1:size(tX, 2), [regressionSpuriousVariables 34]+1)) X_train(:, 18).^3 myPoly(X_train(:, 34), 6)];
[rmseTr, rmseTe] = genericKCV( y_train, txExt4,...
@leastSquares, @rmse, [], 4, [], 100);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using transforms on 18th and 34th while also removing spurious ones, gives test RMSE: ' num2str(meanErrTe)]);

txExt4 = [tX(:, setdiff(1:size(tX, 2), union([regressionSpuriousVariables 34], catVars)+1)) X_train(:, 18).^3 myPoly(X_train(:, 34), 6)];
[rmseTr, rmseTe] = genericKCV( y_train, txExt4,...
@leastSquares, @rmse, [], 4, [], 100);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using transforms on 18th and 34th while also removing spurious ones & categoricals, gives test RMSE: ' num2str(meanErrTe)]);

%% preliminary fitting (classification)

% Form (y,tX) to get regression data in matrix form
tX = [ones(N,1) X_train(:, :)];

noOfSeeds = 10;

ranktX = rank(tX);
disp(' ');
disp(['rank of data columns is :' num2str(ranktX)]);

% simple logistic regression
alpha = 0.1;
betaLR = logisticRegression(y_train,tX,alpha);
temp = tX*betaLR;
[LRrmseTr, LRrmseTe] = genericKCV( y_train, tX,...
@logisticRegression, @rmseClassification, [], 4, alpha, noOfSeeds);
LRrmseTe = mean(LRrmseTe);
disp(' ');
disp(['simple Logistic regression gives test RMSE rmse = ' num2str(LRrmseTe)]);

noOfLambdas = 10;
RRrmses = zeros(1, noOfLambdas);
lambdas = logspace(-2, 2, noOfLambdas);

disp(' ');
for i = 1:noOfLambdas

    betaPLR = penLogisticRegression(y_train, tX, alpha, lambdas(i));
    temp = tX*betaPLR;
    RRrmses(i) = rmseClassification(y_train, temp);
    disp(['for lambda = ' num2str(lambdas(i)) ' penalized logistic regression fits with training RMSE ' num2str(RRrmses(i))])
    
end

% % % takin all variables + sqrts of 11th and 24th
% txExt = [tX (X_train(:, 11)-min(X_train(:, 11))).^0.5 (X_train(:, 24)-min(X_train(:, 24))).^0.5];
% [rmseTr, rmseTe] = genericKCV( y_train, txExt,...
% @logisticRegression, @rmseClassification, [], 4, alpha, noOfSeeds);
% meanErrTe = mean(rmseTe);
% disp(' ')
% disp(['estimated test error using sqrts of 11th and 24th variables, gives test RMSE: ' num2str(meanErrTe)]);
% 
% %taking some degrees (2nd)
% txExt = [tX  X_train(:, 11).^2 X_train(:, 24).^2];
% [rmseTr, rmseTe] = genericKCV( y_train, txExt,...
% @logisticRegression, @rmseClassification, [], 4, alpha, noOfSeeds);
% meanErrTe = mean(rmseTe);
% disp(' ')
% disp(['estimated test error using squares of 11th and 24th variables, gives test RMSE: ' num2str(meanErrTe)]);

% %taking sqrts + some degrees
txExt1 = [tX (X_train(:, 11)-min(X_train(:, 11))).^0.5 (X_train(:, 24)-min(X_train(:, 24))).^0.5 X_train(:, 11).^2 X_train(:, 24).^2];
[rmseTr, rmseTe] = genericKCV( y_train, txExt1,...
@logisticRegression, @rmseClassification, [], 4, alpha, noOfSeeds);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using sqrts of 11th and 24th variables + squares, gives test RMSE: ' num2str(meanErrTe)]);

% % %taking everything except categorical variables
% txExt = tX(:, setdiff(1:(D+1), catVars));
% [rmseTr, rmseTe] = genericKCV( y_train, txExt,...
% @logisticRegression, @rmseClassification, [], 4, alpha, noOfSeeds);
% meanErrTe = mean(rmseTe);
% disp(' ')
% disp(['estimated test error when removing all categorical variables, gives test RMSE: ' num2str(meanErrTe)]);

%taking sqrts + 2nd and 3rd degrees
txExt2 = [tX (X_train(:, 11)-min(X_train(:, 11))).^0.5 (X_train(:, 24)-min(X_train(:, 24))).^0.5...
    X_train(:, 11).^2 X_train(:, 24).^2 X_train(:, 11).^3 X_train(:, 24).^3];
[rmseTr, rmseTe] = genericKCV( y_train, txExt2,...
@logisticRegression, @rmseClassification, [], 4, alpha, noOfSeeds);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using sqrts of 11th and 24th variables + squares + 3rd degree, gives test RMSE: ' num2str(meanErrTe)]);

% txExt = [tX(:, setdiff(1:(D+1), catVars)) dummyCategoricalVariables];
% [rmseTr, rmseTe] = genericKCV( y_train, txExt,...
% @logisticRegression, @rmseClassification, [], 4, alpha, noOfSeeds);
% meanErrTe = mean(rmseTe);
% disp(' ')
% disp(['estimated test error using dummy variables, gives test RMSE: ' num2str(meanErrTe)]);