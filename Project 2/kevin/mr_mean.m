
splitData;
%%

m = computeMeanPerArtist(Ytrain_new);
% repmat on m

repm = repmat(m,size(Ytrain_new,1),1);

% Don't take into account the 0 count
idx = find(Ytest_weak >0);

%%
rmseTr = rmse(Ytest_weak(idx),repm(idx))
%rmseTr = sqrt(2*computeCost(Ytest_weak,repm))