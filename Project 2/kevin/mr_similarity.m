%% Load file
close all; clear all;
%load('songTrain.mat');
splitData;
%Ytrain(Ytrain>0) = 1;
ua_tr = Ytrain_new; % users-artists 
ua_te = Ytest_weak;
S = artistName;


rmse = wrapperCF(ua_tr,ua_te,S);

%% Are the friends similar?
idx1 = find(Gtrain==1);
simOfFriends = Su(idx1);
meanSimilarity = mean(simOfFriends)

