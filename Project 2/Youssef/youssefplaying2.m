% youssef's playing with the data script, project 2

% clear all;
% close all;
% clc

projectDir = '/users/youssef/Documents/Matlab/PCML/Projects/Project 2';
addpath(genpath(projectDir));
cd(projectDir);

% Load both features and training images
load train_feats;
%load train_imgs;
N = length(labels);

% Generate feature vectors (so each one is a row of X)
disp('Generating feature vectors..');
D = numel(feats{1});  % feature dimensionality
X = zeros([N D]);

for i=1:N
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end

%% Fitting using (penalized)  logistic regression - parameters + normalization

disp('normalizing data and building tX matrix');
[normX, ~, ~] = zscore(X); % train, get mu and std

tX = [ones(size(normX, 1), 1) normX];
y = labels;
y(y == -1) = 0; % logistic regression needs the classes to be binary

noOfLambdas = 5;
lambdas = logspace(-3, 2, noOfLambdas);
noOfSeeds = 10;
alpha = 0.1;
K = 4;

%% Fitting using  logistic regression - simple run

betaLR = logisticRegression(y,tX,alpha);
temp = tX*betaLR;
logLossError = logLoss(y, probEstimate(temp));
disp(['log loss train error with simple logistic regression is' num2str(logLossError)])

%% Fitting using penalized logistic regression - using KCV

[rmseTr, rmseTe] = genericKCV( labels, tX,...
@penLogisticRegression, @logLossWrapper, [], K, alpha, noOfSeeds);
meanErrTe = mean(rmseTe);
disp(' ')
disp(['estimated test error on PD using penalized Logistic Regression: ' num2str(meanErrTe)]);

%% Training neural networks - data splitting
disp('Splitting into train/test..');
% NOTE: you should do this randomly! and k-fold!
Tr.idxs = 1:2:size(X,1);
Tr.X = X(Tr.idxs,:);
Tr.y = labels(Tr.idxs);

Te.idxs = 2:2:size(X,1);
Te.X = X(Te.idxs,:);
Te.y = labels(Te.idxs);

%% Training neural networks - parameters

dimensions = [size(Tr.X,2) 10 2];
batchSize = 100;
plotFlag = 1;
learningRate = 2;
noEpochs = 25;

%% Training neural networks - normalizing data

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = batchSize * floor( size(Tr.X) / batchSize);
Tr.X = Tr.X(1:numSampToUse,:);
Tr.y = Tr.y(1:numSampToUse);

% normalize data
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data


%% Training neural networks - simple NN
[ trainedNN1 ] = trainNewBinaryNN(Tr.normX, Tr.y, dimensions, noEpochs, ...
    batchSize, plotFlag, learningRate, [], [] );
nnPred = predictNNBinaryOutput(Te.normX, trainedNN1 );

%% Training neural networks - simple NN + dropout

dropout = 0.3;
[ trainedNN2 ] = trainNewBinaryNN(Tr.normX, Tr.y, dimensions, noEpochs, ...
    batchSize, plotFlag, learningRate, dropout, [] );
nnPredWithDropout = predictNNBinaryOutput(Te.normX, trainedNN2 );

%% Training neural networks - simple NN + L2 regularization

lambda = 1/1000;
[ trainedNN3 ] = trainNewBinaryNN(Tr.normX, Tr.y, dimensions, noEpochs, ...
    batchSize, plotFlag, learningRate, [], lambda );
nnPredWithReg = predictNNBinaryOutput(Te.normX, trainedNN3 );

%% comparing results

% let's also see how random predicition does
randPred = rand(size(Te.y))*2-1;

% and plot all together, and get the performance of each method

% this is to show it in the legend
methodNames = {'NN with L2 reg = 1/50', 'NN with dropout = 0.3', 'Neural Network (NN)', 'Random'};

avgTPRList = evaluateMultipleMethods( Te.y > 0, [nnPredWithReg, nnPredWithDropout, nnPred,randPred], true, methodNames );
