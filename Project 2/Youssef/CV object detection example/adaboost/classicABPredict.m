% if outputWeightedWeakLearnerMatrix = true, outputs a matrix instead of
% scores. default = false
function ret = classicABPredict( model, feats, maxIter, outputWeightedWeakLearnerMatrix )

if nargin < 4
    outputWeightedWeakLearnerMatrix = false;
end

if nargin < 3 || isempty(maxIter)
    maxIter = length(model.WL);
end

if maxIter > length(model.WL)
    error('maxIter exceeds number of trained weak learners');
end

if outputWeightedWeakLearnerMatrix
    scores = zeros([size(feats,1), maxIter]);
    
    for I=1:maxIter
        scores(:,I) = model.WL(I).inv * model.alpha(I) * ((feats(:, model.WL(I).col) >= model.WL(I).thr) * 2 - 1);
    end
else
    scores = zeros([size(feats,1), 1]);
    
    for I=1:maxIter
        scores = scores + model.WL(I).inv * model.alpha(I) * ((feats(:, model.WL(I).col) >= model.WL(I).thr) * 2 - 1);
    end
end


ret = scores;