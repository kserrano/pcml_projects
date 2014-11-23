% retunrns the point of best VOC score value and the corresponding
% threshold
% trainImgs: list of training images + labels
% scoreImg: your proposed image that measures how synapse-like every voxel
% is
function [threshold, accuracy] = findBestThreshold( trainLabels, scoreImg )

minVal = min(scoreImg(:));
maxVal = max(scoreImg(:));

thresholds = linspace( minVal, maxVal, 500 );

inv = [];
normal = [];

[inv.threshold, inv.accuracy] = findBestThresholdAux( trainLabels, scoreImg, thresholds, true );
[normal.threshold, normal.accuracy] = findBestThresholdAux( trainLabels, scoreImg, thresholds, false );

if inv.accuracy > normal.accuracy
    threshold = inv.threshold;
    accuracy = inv.accuracy;
    fprintf('Best accuracy found when inverting the score image\n');
else
    threshold = normal.threshold;
    accuracy = normal.accuracy;
    fprintf('Best accuracy found not inverting the score image\n');
end

fprintf('Accuracy: %.2f, Threshold: %.2f\n', accuracy, threshold);

function [threshold, accuracy] = findBestThresholdAux( trainLabels, scoreImg, thresholds, invert )

accList = zeros([length(thresholds) 1]);

trainLabels = trainLabels(:) > 0;

if ~invert
    for I=1:length(thresholds)

        thr = thresholds(I);
        
        tScore = (scoreImg(:) >= thr);
        
        TP = sum(tScore & trainLabels);
        FP = sum(tScore & (~trainLabels));
        FN = sum((~tScore) & (trainLabels));

        accList(I) = TP / (TP + FP + FN);
    end
else
    for I=1:length(thresholds)

        thr = thresholds(I);
        tScore = (scoreImg(:) < thr);
        
        TP = sum(tScore & trainLabels);
        FP = sum(tScore & (~trainLabels));
        FN = sum((~tScore) & (trainLabels));

        accList(I) = TP / (TP + FP + FN);
    end
end

[accuracy, idx] = max(accList);
threshold = thresholds(idx);