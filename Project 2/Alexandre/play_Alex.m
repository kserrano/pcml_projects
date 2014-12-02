%% Play with MR
% Find indices of 1s in the sparse matrix

clear all
close all
load songTrain
s = nonzeros(Ytrain);
maxCount = max(s);
histogram(s);
h = hist(s,10000);

%% -- Normalize the data

Ynorm = bsxfun(@rdivide,Ytrain,std(Ytrain));

%% -- PCA with MR
[coeff,score,latent,tsquared,explained,mu] = pca(full(Ytrain));

%% -- SVD with MR
[U,Sigma,V] = svd(full(Ytrain),'econ');
s = diag(Sigma); % Vector of singular values
normsqS = sum(s.^2);
plot(cumsum(s.^2)/normsqS,'x')  % Cumulative fraction of variance explained
xlabel('Mode k')
ylabel('Cumulative Variance fraction explained')
ylim([0 1])

%% Play with PD

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
[Xtr, yTr, XTe, yTe] = split(y,X,0.8);


%% -- Do the kmeans algorithm
[idx] = kmeans(Xtr,2);

%% -- Change the classification 

idx(idx == 2) = 1;
idx(idx == 1) = -1;

%% Compute the rmse and number of errors

rmse=sqrt(sum((yTr(:)-idx(:)).^2)/numel(yTr));
noOfError = nnz(abs(yTr-idx));
rateOfError = ((noOfError / numel(yTr))*100);




