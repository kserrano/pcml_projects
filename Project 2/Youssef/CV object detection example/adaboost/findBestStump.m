function [bestFeat, bestThr, bestInv, bestErr] = findBestStump( feats, labels, w )

fMin = min(feats);
fMax = max(feats);

bestErr = inf;
bestThr = 0;
bestFeat = 0;
bestInv = 0;

for F=1:size(feats,2)
    
    %thrList = linspace(fMin(F),fMax(F),100);
    thrList = fMin(F) + rand([100,1]) * (fMax(F) - fMin(F));
    
    for T=1:length(thrList)
        
        thr = thrList(T);
        
        normErr = sum( w .* ((feats(:,F) > thr) ~= labels) );
        invErr = 1 - normErr;
        
        if bestErr > normErr
            bestThr = thr;
            bestFeat = F;
            bestInv = 1;
            bestErr = normErr;
        end
        
        if bestErr > invErr
            bestThr = thr;
            bestFeat = F;
            bestInv = -1;
            bestErr = invErr;
        end
    end
end
