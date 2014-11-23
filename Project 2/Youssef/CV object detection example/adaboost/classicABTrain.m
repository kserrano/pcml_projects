% classic adaboost training function
function [model, allWeights] = classicABTrain( feats, labels, numIters )


useGentleAB = false;



balanceClasses = false;



randomAlpha = false;


applySubSampAfter = [];
stopReweightingAfter = [];



shrinkageFactor = 1.0;



subSampFactor = 0.1;


if size(feats,1) ~= numel(labels)
    error('Features and label size do not match');
end

if any(unique(labels) ~= [-1 1]')
    error('Labels must be -1 or 1 exclusively');
end

% initialize weights
w = ones(size(labels));

if nargout > 1
    allWeights = zeros([length(w) numIters]);
end

if balanceClasses
    fprintf('Balancing classes\n');
    w(labels > 0) = sum( labels < 0 );
    w(labels < 0) = sum( labels > 0 );
    
    pIdx = find(labels > 0);
    nIdx = find(labels < 0 );
    
    fprintf('PosW: %f, NegW: %f\n', w(pIdx(1)), w(nIdx(1)));
    fprintf('NumPos: %d, NumNeg: %d\n', sum(labels > 0), sum(labels < 0) );
end

w = w / sum(w(:));

model.alpha = [];
model.WL = [];

scores = zeros([size(feats,1) 1]);

posIdxs = find(labels > 0);
negIdxs = find(labels < 0);

for I=1:numIters
    
    allWeights(:,I) = w;
    
    subsampled = false;
    wupd = false;
    

        subsampled = true;
        
        % get negative ones
        samplingFactor = 2;
        newNegIdxs = randsample( negIdxs, samplingFactor * length(posIdxs), true, w(negIdxs) );

        % find best weak learner
        subW = zeros([length(posIdxs)+length(newNegIdxs) 1]);
        subW(1:length(posIdxs)) = w(posIdxs);
        subW(length(posIdxs)+1:end) = sum(w(negIdxs)) / length(newNegIdxs);
        
        ssIdxs = [posIdxs; newNegIdxs];

        [idx, thr, inv, ~]=findBestStump( feats(ssIdxs,:), uint8(labels(ssIdxs) > 0), subW );

        predLbl = inv * ((feats(:, idx) >= thr) * 2 - 1);

        err = sum(w .* ( (predLbl > 0) ~= (labels > 0) ));
    
    % find alpha
    model.alpha(I) = 0.5 * log( (1 - err) / err );
    
    model.alpha(I) = shrinkageFactor * model.alpha(I);
    
    if (useGentleAB)
        model.alpha(I) = 1.0;
    end
    
    if (randomAlpha)
        model.alpha(I) = rand() * model.alpha(I);
    end
    
    % set weak learner
    model.WL(I).col = idx;
    model.WL(I).thr = thr;
    model.WL(I).inv = inv;
    
    if isempty(stopReweightingAfter) || (I < stopReweightingAfter)
        % update weights
        expAlpha = exp(model.alpha(I));
        expMAlpha = exp(-model.alpha(I));

        % 0 or 1 for each sample
        T = ((feats(:, model.WL(I).col) >= thr) * 2 - 1)*inv;
        D = T == labels;
        w(D==1) = w(D==1) * expMAlpha;
        w(D==0) = w(D==0) * expAlpha;
        
        % re-normalize
        w = w / sum(w(:));
        
        wupd = true;
    end
    
    scores = scores + T * model.alpha(I);
    
    fprintf('Iter %d / %d, error %.2f%% / %d %d\n', I, numIters, 100 * sum((scores > 0) ~= (labels > 0)) / length(labels), subsampled, wupd );
end