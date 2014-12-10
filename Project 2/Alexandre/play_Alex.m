%% Play with MR
% Find indices of 1s in the sparse matrix

clear all
close all
load songTrain
s = nonzeros(Ytrain);
maxCount = max(s);
histogram(s);
h = hist(s,3500);

%% -- Normalize the data

% V?rifier les lignes ... pour les utilisateurs. Voir script de Youssef.
Ynorm = bsxfun(@rdivide,Ytrain',std(Ytrain'));
Ynorm = Ynorm';

%% -- PCA with MR
[coeff,score,latent,tsquared,explained,mu] = pca(full(Ynorm));
Ynew = sparse(score)*sparse(coeff');

%% -- SVD with MR

% Normalize the data
[U,Sigma,V] = svd(full(Ynorm),'econ');
s = diag(Sigma); % Vector of singular values
normsqS = sum(s.^2);
plot(cumsum(s.^2)/normsqS,'x')  % Cumulative fraction of variance explained
xlabel('Mode k')
ylabel('Cumulative Variance fraction explained')
ylim([0 1])

%% Play with PD

clear all
close all

% Load both features and training images
load train_feats;
load train_imgs;
N = length(imgs);


%% -- Generate feature vectors (so each one is a row of X)
fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
X = zeros([length(imgs) D]);

for i=1:length(imgs)
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end

y = labels;

%% -- Example: split half and half into train/test randomly 80%, 20%
[Xtr, yTr, XTe, yTe] = split(y,X,0.9);



%% -- Random Split

disp('Splitting into train/test..');
% NOTE: you should do this randomly! and k-fold!
Tr.idxs = 1:2:size(X,1);
Tr.X = X(Tr.idxs,:);
Tr.y = labels(Tr.idxs);

Te.idxs = 2:2:size(X,1);
Te.X = X(Te.idxs,:);
Te.y = labels(Te.idxs);

%% -- Data Normalization
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data


%% -- Reduce dimensionality using PCA

[Te.Xres, Te.Xred] = pcares(Te.normX, (size(Te.normX,1) - 1));
Te.Xred = Te.Xred(:,1:((size(Te.normX,1)-1)));

%% -- Do the soft-kmeans algorithm 

obj = fitgmdist(Te.Xred,2,'RegularizationValue',0.001);
[idxSoft,nlogl,P] = cluster(obj,Te.Xred);

idxSoft(idxSoft == 2) = 1;
idxSoft(idxSoft == 1) = -1;

SoftKmeansPredict = idxSoft;

%% -- Do the kmeans algorithm
[idx,C,sumd,D] = kmeans(Te.normX,2);

idx(idx == 2) = 1;
idx(idx == 1) = -1;

kmeansPredict = idx;

%% comparing results

% let's also see how random predicition does
randPred = rand(size(Te.y))*2-1;

% and plot all together, and get the performance of each method

% this is to show it in the legend
methodNames = {'Soft k-means','k-means', 'Random'};



avgTPRList = evaluateMultipleMethods( Te.y > 0, [SoftKmeansPredict, kmeansPredict, randPred], true, methodNames );


