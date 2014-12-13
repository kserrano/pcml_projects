%% initialize 
clear all;
load songTrain;
load songTestPairs;
Ytest_weak_pred = Ytest_weak_pairs;
Ytest_strong_pred = Ytest_strong_pairs;
percent = 10;
i = 1;
%% 

[Ytrain_new]=splitDataMRmean(Ytrain,Gtrain,percent/100);
    
m = computeMeanPerArtist(Ytrain_new);
% repmat on m
m(m==0) = 1;
repm_weak = repmat(m,size(Ytest_weak_pairs,1),1);
repm_strong = repmat(m,size(Ytest_strong_pairs,1),1);
idxw = find(Ytest_weak_pairs>0);
idxs = find(Ytest_strong_pairs>0);
Ytest_weak_pred(idxw) = repm_weak(idxw);
Ytest_strong_pred(idxs) = repm_strong(idxs);
rmse = sqrt(2*computeCost(Ytest_weak_pairs,Ytest_weak_pred))    
%%
save('songPred', 'Ytest_strong_pred', 'Ytest_weak_pred');