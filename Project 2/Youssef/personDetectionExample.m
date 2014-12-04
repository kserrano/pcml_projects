clear all;
clc
close all

% -- GETTING STARTED WITH THE PERSON DETECTION DATASET -- %
% IMPORTANT: Make sure you downloaded Piotr's toolbox: http://vision.ucsd.edu/~pdollar/toolbox/doc/
%            and add it to the path with
%            addpath(genpath('where/the/toolbox/is'))
%
%    And make sure you downloaded the train_feats.mat and train_imgs.mat
%    files we provided you with.

% add path to piotr's toolbox

% ---------------------------------------------
% ====>   FIX TO YOUR ACTUAL TOOLBOX PATH <====
% ---------------------------------------------
%addpath(genpath('../toolbox/'));

% Load both features and training images
load train_feats;
load train_imgs;
N = length(imgs);

%% --browse through the images, and show the feature visualization beside
%  -- You should explore the features for the positive and negative
%  examples and understand how they resemble the original image.
for i=1:10
    clf();
   
    chosenImageIndex = floor(rand*N);
    
    subplot(121);
    imshow(imgs{chosenImageIndex}); % image itself
    
    subplot(122);
    imagesc( hogDraw(feats{chosenImageIndex}) ); colormap gray;
    axis off; colorbar off;
    
    pause;  % wait for keydo that then, 
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

%%
disp('Training simple neural network..');



[ trainedNN ] = trainNewNN( dataInput, dataOutput, dimensions, noEpochs, batchSize, plotFlag, learningRate );

Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

predictNNBinaryOutput(Te.X, nn )

%% with dropout

nn.learningRate = 1; % otherwise the cost function doesn't converge

nn.dropoutFraction = 0.5;
[nn, L] = nntrain(nn, Tr.normX, LL, opts);

nn.testing = 1;
nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPredWithDropout = nn.a{end};
nnPredWithDropout = nnPredWithDropout(:,1) - nnPredWithDropout(:,2);

%% See prediction performance
disp('Plotting performance..');
% let's also see how random predicition does
randPred = rand(size(Te.y));

% and plot all together, and get the performance of each
methodNames = {'Neural Network with dropout 0.5', 'Neural Network', 'Random'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods( Te.y > 0, [nnPredWithDropout, nnPred,randPred], true, methodNames );

% now you can see that the performance of each method
% is in avgTPRList. You can see that random is doing very bad.
avgTPRList

%% visualize samples and their predictions (test set)
figure;
for i=20:30  % just 10 of them, though there are thousands
    clf();
    
    subplot(121);
    imshow(imgs{Te.idxs(i)}); % image itself
    
    subplot(122);
    imagesc( hogDraw(feats{Te.idxs(i)}) ); colormap gray;
    axis off; colorbar off;
    
    % show if it is classified as pos or neg, and true label
    title(sprintf('Label: %d, Pred: %d', labels(Te.idxs(i)), 2*(nnPred(i)>0) - 1));
    
    pause;  % wait for keydo that then, 
end
