function [ avgTPRList ] = wrapperSVM( Tr,Te )
%WRAPPERSVM Summary of this function goes here
%   Detailed explanation goes here
%% create the model
fprintf('Creating the model..\n');
SVMModelTr = fitcsvm(Tr.X,Tr.y);

%% predict new data
fprintf('Predict new data...\n');
[predlabels, scores] = predict(SVMModelTr,Te.X);


%% count correctness
count = sum(predlabels==Te.y)/length(Te.y)
%%
sc = scores(:,1) + scores(:,2);
%% compute TPR
% and plot all together, and get the performance of each

methodNames = {'SVM?','SVM'}; % this is to show it in the legend
avgTPRList = evaluateMultipleMethods( Te.y > 0, [scores], true, methodNames );


end

