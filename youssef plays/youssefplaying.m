% Comments and observations made by youssef

clc
clear all
close all

%load('MexicoCity_regression.mat')
load('MexicoCity_classification.mat')

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

%% normalizing the features

% for regression variables
% normalize features (store the mean and variance)
meanX = mean(X_train);

% 42 first features seem to be gaussian distributed, can be normalized
normInputVars = 1:33;

% normalizing the input variables
X_train(:, normInputVars) = X_train(:, normInputVars) - repmat(meanX(normInputVars), N, 1);
stdX = std(X_train);
X_train(:, normInputVars) = X_train(:, normInputVars)./repmat(stdX(:, normInputVars), N, 1);

% normalizing the output variable
y_train = (y_train-mean(y_train))/std(y_train);

% Form (y,tX) to get regression data in matrix form
tX = [ones(N,1) X_train(:, :)];

%% correlation & normal distribution analysis

statistics = [y_train X_train];

[ spearmanCoeffs, spearmanSig, pearsonCoeffs, pearsonSig,...
    normalityTests, corrExistence, corrSuitability] = ...
    computeCorrelationCoeffs( statistics, D+1);

normalityTests = circshift(normalityTests, -1);

figure;
stem(spearmanSig.*spearmanCoeffs)
title('correlated variables with coeff, 0 otherwise, spearman')

figure;
stem(pearsonSig.*pearsonCoeffs)
title('correlated variables with coeff, 0 otherwise, pearson')

%% preliminary fitting (regression)

% simple least squares
betaLS = leastSquares(y_train, tX);
y_hatLS = tX*betaLS;
LSrmse = rmse(y_hatLS, y_train);
disp(' ');
disp(['simple LS gives rmse = ' num2str(LSrmse)]);

alpha = 0.01;
betaLSGD = leastSquaresGD(y_train, tX, alpha);
y_hatLSGD = tX*betaLSGD;
LSGDrmse = rmse(y_hatLSGD, y_train);
disp(' ');
disp(['gradient LS gives rmse = ' num2str(LSGDrmse)]);

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

%% preliminary fitting (classification)

% Form (y,tX) to get regression data in matrix form
tX = [ones(N,1) X_train(:, :)];

% simple logistic regression
alpha = 0.1;
betaLR = logisticRegression(y_train,tX,alpha);
temp = tX*betaLR;
y_hatLS = temp > 0;
LSrmse = rmse(y_train, y_hatLS);
disp(' ');
disp(['simple Logistic regression gives rmse = ' num2str(LSrmse)]);

noOfLambdas = 10;
RRrmses = zeros(1, noOfLambdas);
lambdas = logspace(-2, 2, noOfLambdas);

disp(' ');
for i = 1:noOfLambdas

    betaPLR = penLogisticRegression(y_train, tX, alpha, lambdas(i));
    temp = tX*betaPLR;
    y_hatPLR = temp > 0;
    RRrmses(i) = rmse(y_train, y_hatPLR);
    disp(['for lambda = ' num2str(lambdas(i)) ' ridge regression fits with RMSE ' num2str(RRrmses(i))])
    
end
