% youssef's playing with the data script, project 2

clear all;
close all;
clc

projectDir = '/users/youssef/Documents/Matlab/PCML/Projects/Project 2';
addpath(genpath(projectDir));
rmpath([projectDir '/Piotr toolbox'])
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

y = labels;

%% Fitting using (penalized)  logistic regression - parameters

disp('normalizing data and building tX matrix');
[normX, ~, ~] = zscore(X); % train, get mu and std

%tX = [ones(size(normX, 1), 1) normX];
y(y == -1) = 0; % logistic regression needs the classes to be binary

noOfLambdas = 10;
lambdas = logspace(-3, 2, noOfLambdas);
noOfSeeds = 10;
alpha = 0.1;
K = 4;

%% Data splitting - RUN FOR BOTH LR and NN

disp('Splitting into train/test..');
% NOTE: you should do this randomly! and k-fold!
Tr.idxs = 1:2:size(X,1);
Tr.X = X(Tr.idxs,:);
Tr.y = y(Tr.idxs);

Te.idxs = 2:2:size(X,1);
Te.X = X(Te.idxs,:);
Te.y = y(Te.idxs);

%% Fitting using  logistic regression - preparations
Tr.tX = [ones(size(Tr.normX, 1), 1) Tr.normX];
Te.tX = [ones(size(Te.normX, 1), 1) Te.normX];

%% Fitting using logistic regression - simple run


betaLR = logisticRegression(Tr.y,Tr.tX,alpha);
temp = Te.tX*betaLR;
LRPred = probEstimate(temp)*2-1;

LRTPR = TPRs(Te.y, LRPred);

%% Fitting using penalized logistic regression - simple run

lambda = 44;
betaPLR = penLogisticRegression(Tr.y,Tr.tX,alpha,lambda);
temp2 = Te.tX*betaPLR;
PLRPred = probEstimate(temp2)*2-1;
PLRTPR = TPRs(Te.y, PLRPred);

%% Fitting using penalized logistic regression - using KCV

[logLossTr, logLossTe] = genericKCV( Tr.y, Tr.tX,...
@penLogisticRegression, @TPRsWrapper, lambdas, K, alpha, noOfSeeds);

[bestParameter, achievedErrorTe ] = analyzeKCVresults( logLossTe,...
    logLossTr, 'lambda', lambdas, 'Penalized Logistic regression on PD data', 'log loss' );

%% Training neural networks - parameters

dimensions = [size(Tr.X,2) 10 2];
batchSize = 100;
plotFlag = 1;
learningRate = 2;
noEpochs = 25;

%% Training neural networks - further preparations after splitting

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.

numSampToUse = batchSize * floor( size(Tr.X) / batchSize);
%numSampToUse = batchSize;
Tr.X = Tr.X(1:numSampToUse,:);
Tr.y = Tr.y(1:numSampToUse);

%% normalizing data - RUN FOR BOTH LR and NN

[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data


%% Training neural networks - simple NN
rand('seed',8339);
[ trainedNN1 ] = trainNewBinaryNN(Tr.normX, Tr.y, dimensions, noEpochs, ...
    batchSize, plotFlag, learningRate, [], [], [] );
nnPred = predictNNBinaryOutput(Te.normX, trainedNN1 );

%% Training neural networks - sigmoid inner activations

rand('seed',8339);
[ trainedNN1 ] = trainNewBinaryNN(Tr.normX, Tr.y, dimensions, noEpochs, ...
    batchSize, plotFlag, learningRate, [], [], 'sigm' );
nnPredSigm = predictNNBinaryOutput(Te.normX, trainedNN1 );

%% Training neural networks - simple NN + dropout
rand('seed',8339);
dropout = 0.3;
[ trainedNN2 ] = trainNewBinaryNN(Tr.normX, Tr.y, dimensions, noEpochs, ...
    batchSize, plotFlag, learningRate, dropout, [], [] );
nnPredWithDropout = predictNNBinaryOutput(Te.normX, trainedNN2 );

%% Training neural networks - simple NN + L2 regularization
rand('seed',8339);
lambda = 1/1000;
[ trainedNN3 ] = trainNewBinaryNN(Tr.normX, Tr.y, dimensions, noEpochs, ...
    batchSize, plotFlag, learningRate, [], lambda, [] );
nnPredWithReg = predictNNBinaryOutput(Te.normX, trainedNN3 );

%% Training neural networks - simple NN + dropout + L2 regularization
rand('seed',8339);
[ trainedNN4 ] = trainNewBinaryNN(Tr.normX, Tr.y, dimensions, noEpochs, ...
    batchSize, plotFlag, learningRate, dropout, [], 'sigm' );
nnPredWithDropoutAndSigm = predictNNBinaryOutput(Te.normX, trainedNN4 );

%% Training neural networks - final dropout + L2 regularization + sigm

rand('seed',8339);

dimensions = [size(Tr.X,2) 50 25 10 2];
lambda = 10^-6;
dropout = 0.3;

[ trainedNN5 ] = trainNewBinaryNN(Tr.normX, Tr.y, dimensions, noEpochs, ...
    batchSize, plotFlag, learningRate, dropout, [], 'sigm' );
nnPredWithDropoutAndSigm = predictNNBinaryOutput(Te.normX, trainedNN5 );

%% comparing results
rand('state',8339);
% let's also see how random predicition does
randPred = rand(size(Te.y))*2-1;

% and plot all together, and get the performance of each method

% this is to show it in the legend
methodNames = {'NN with dropout = 0.3 and sigmoid', 'NN with L2 reg = 1/1000', 'NN with dropout = 0.3', 'NN with sigmoid activ.' 'Neural Network (NN)', 'Random'};

avgTPRList = evaluateMultipleMethods( Te.y > 0, [nnPredWithDropoutAndSigm, nnPredWithReg, nnPredWithDropout, nnPredSigm, nnPred,randPred], true, methodNames );
