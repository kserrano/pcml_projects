%% initialize 
clear all;
load songTrain;
load songTestPairs;
Ytest_weak_pred = full(Ytest_weak_pairs);
Ytest_strong_pred = full(Ytest_strong_pairs);

Ytest_weak_pairs = full(Ytest_weak_pairs);
Ytest_strong_pairs = full(Ytest_strong_pairs);


Ytrain = full(Ytrain);
Gtrain = full(Gtrain);

Ytest = Ytrain(Ytest_weak_pairs);

percent = 10;
i = 1;
%% 

[Ytrain_new]=splitDataMRmean(Ytrain,Gtrain,percent/100);
    
m = computeMeanPerArtist(Ytrain_new);
% repmat on m
m(m==0) = 1;
repm_weak = full(repmat(m,size(Ytest_weak_pairs,1),1));
repm_strong = full(repmat(m,size(Ytest_strong_pairs,1),1));
idxw = Ytest_weak_pairs>0;
idxs = Ytest_strong_pairs>0;
repm_weak = exp(repm_weak);
repm_weak(~idxw) = 0;

% Ytest_strong_pred(idxs) = repm_strong(idxs);

rmse = rmse(Ytest_weak_pred, Ytest_weak_pairs);
%%
save('songPred', 'Ytest_strong_pred', 'Ytest_weak_pred');