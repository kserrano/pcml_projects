% script for cross-validating SVM

% preparing data
clear all;
close all;
clc

projectDir = '/Users/alexandrehelfre/Dropbox/MATLAB/pcml_projects/Project 2';
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

normalizeFlag = 1;
K = 4;
noOfSeeds = 6;
constantSVMoptions.kernel = 'rbf';

%% KCVating the kernel scale (sigma for rbf)

%choose the no of sigmas carefully, with the noOfSeeds & K in mind, because
% the code is going to train once and predict twice (test, train)
% noOfSeeds*K*noOfSigmas models. For rbf kernels this is going to take
% WHILE (test to see how much 1 model takes to train, if it's t minutes than the
% code is going to take t*noOfSeeds*K*noOfSigmas/60 hours !! For 10 minutes
% with the current parameters if it's 10 minutes per model it's 10*3*4*4/60
% = 8 hours. Measure time and change parameters yourself. 

%This is with the old generic KCV function, the updated one using parallel
%pools is faster, but no guarantee on how much faster, though probably a
%lot.

noOfSigmas = 6;
% sigmas = logspace(-1, 3, noOfSigmas);
sigmas = logspace(30, 200, noOfSigmas);
constantSVMoptions.C = []; % indicates that this isn't the parameter to genericKCV

[errorsTr, errorsTe] = genericKCV( y, X,...
   @trainSVM, @predictSVMoutput, @TPRs, sigmas, K, [], noOfSeeds,...
   [], constantSVMoptions, normalizeFlag);

% it saves the results directly, these are to be shared
save('SVM KCV results (light) - focalized','errorsTr', 'errorsTe', 'noOfSeeds', 'K', 'sigmas', 'constantSVMoptions');