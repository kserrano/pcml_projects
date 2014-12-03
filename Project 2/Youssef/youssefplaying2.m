% youssef's playing with the data script, project 2

clear all;
close all;
clc

addpath(genpath('/users/youssef/Documents/Matlab/PCML/Projects/Project 2'));

% Load both features and training images
load train_feats;
load train_imgs;
N = length(imgs);

%% Generate feature vectors (so each one is a row of X)
disp('Generating feature vectors..');
D = numel(feats{1});  % feature dimensionality
X = zeros([length(imgs) D]);

for i=1:length(imgs)
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end

%% -- Example: split half and half into train/test
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

%% Training neural networks - simple NN
[ trainedNN ] = trainNewNN( dataInput, dataOutput, dimensions, noEpochs, batchSize, plotFlag, learningRate );

