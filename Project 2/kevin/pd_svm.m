
%% Load
close all;
clear all;

load('train_feats.mat');
load('train_imgs.mat');

%% 
% feats = feats(1:100,1);
% labels = labels(1:100,1);
% imgs = imgs(1:100,1);
%% -- Generate feature vectors (so each one is a row of X)
fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
X = zeros([length(imgs) D]);

for i=1:length(imgs)
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end

%% -- Example: split half and half into train/test
fprintf('Splitting into train/test..\n');
% NOTE: you should do this randomly! and k-fold!
Tr.idxs = 1:2:size(X,1);
Tr.X = X(Tr.idxs,:);
Tr.y = labels(Tr.idxs);

Te.idxs = 2:2:size(X,1);
Te.X = X(Te.idxs,:);
Te.y = labels(Te.idxs);

%% create the model
fprintf('Creating the model..\n');
SVMModelTr = fitcsvm(Tr.X,Tr.y);

%% predict new data
fprintf('Predict new data...\n');
[predlabels, scores] = predict(SVMModelTr,Te.X);


%% count correctness
count = sum(predlabels==Te.y)/length(Te.y)
%%
sc = scores(:,1) + scores(:,2);
%% compute TPR
% and plot all together, and get the performance of each

methodNames = {'SVM?','SVM'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods( Te.y > 0, [scores], true, methodNames );
avgTPRList
%% Compute ROC ?