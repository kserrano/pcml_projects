function [avgTPRList] = wrapperSKM(Tr,Te)

%% Do PCA to reduce the dimensionality (Use the PCA matlab function)
[Te.Xres, Te.Xred] = pcares(Te.X, (size(Te.X,1) - 1));
Te.Xred = Te.Xred(:,1:((size(Te.X,1)-1)));

%% Create the model
fprintf('Creating the model..\n');
GMModel = fitgmdist(Te.Xred,2,'RegularizationValue',0.001);

%% Predict new data
fprintf('Predict new data...\n');

[~,~,softkPredict] = cluster(GMModel,Te.Xred);

%% compute TPR
% and plot all together, and get the performance of each

methodNames = {'SKM?','SKM'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods( Te.y > 0, [softkPredict], true, methodNames );


end


