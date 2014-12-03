
splitData;

m = computeMeanPerArtist(Ytrain);
% repmat on m

repm = repmat(m,size(Ytrain_new,1),1);

% Don't take into account the 0 count
idx = find(Ytrain_new >0);

%%

rmseTr = sqrt(2*computeCostMean(Ytrain_new(idx),repm(idx)))
