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

for i=1:D
    test = unique(X_train(:, i));
    noOfVariableValues(i) = length(test);
end

catVars = find(noOfVariableValues < maxClasses);

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
y_train = (y_train-mean(y_train))/std(y_train);

% Form (y,tX) to get regression data in matrix form
tX = [ones(N,1) X_train(:, :)];

%% correlation & normal distribution analysis

% taking the 18th and 34th variables, and any polynomial transform of them,
% and checking their pearson correlation with the output

txSpecial = [myPoly(X_train(:, 18),5) myPoly(X_train(:, 34),10) ];
statistics = [y_train txSpecial];
% use this line if u want to plot the correlations for all the data instead
%statistics = [y_train X_train];

[ spearmanCoeffs, spearmanSig, pearsonCoeffs, pearsonSig,...
    normalityTests, corrExistence, corrSuitability] = ...
    computeCorrelationCoeffs( statistics, size(statistics, 2));

normalityTests = circshift(normalityTests, -1);

figure;
spearmanGraph = abs(spearmanSig.*spearmanCoeffs);
stem(spearmanGraph)
title('correlated variables with coeff, 0 otherwise, spearman, , in absolute value')

figure;
pearsonGraph = abs(pearsonSig.*pearsonCoeffs);
stem(pearsonGraph)
title('correlated variables with coeff, 0 otherwise, pearson, in absolute value')

figure;
stem(spearmanGraph.*pearsonGraph)
title('correlated variables with coeff, 0 otherwise, pearson x spearman, in absolute value')

figure;
stem(mean([spearmanGraph; pearsonGraph], 1))
title('correlated variables with coeff, 0 otherwise, mean(pearson, spearman), in absolute value')


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

% usung the 18th and 34th variables transforms
txExt =[tX(:, setdiff(1:49, 35)) X_train(:, 18).^3 myPoly(X_train(:, 34), 6)];
[rmseTr, rmseTe] = genericKCV( y_train, txExt,...
@leastSquares, @rmse, [], 4, [], 100);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error using transforms on 18th and 34th, gives test RMSE: ' num2str(meanErrTe)])

%% preliminary fitting (classification)

% Form (y,tX) to get regression data in matrix form
tX = [ones(N,1) X_train(:, :)];

ranktX = rank(tX);
disp(' ');
disp(['rank of data columns is :' num2str(ranktX)]);

% simple logistic regression
alpha = 0.1;
betaLR = logisticRegression(y_train,tX,alpha);
temp = tX*betaLR;
y_hatLS = temp > 0;
LSrmse = rmse(y_train, y_hatLS);
disp(' ');
disp(['simple Logistic regression gives Training rmse = ' num2str(LSrmse)]);

noOfLambdas = 10;
RRrmses = zeros(1, noOfLambdas);
lambdas = logspace(-2, 2, noOfLambdas);

disp(' ');
for i = 1:noOfLambdas

    betaPLR = penLogisticRegression(y_train, tX, alpha, lambdas(i));
    temp = tX*betaPLR;
    y_hatPLR = temp > 0;
    RRrmses(i) = rmse(y_train, y_hatPLR);
    disp(['for lambda = ' num2str(lambdas(i)) ' penalized logistic regression fits with training RMSE ' num2str(RRrmses(i))])
    
end
