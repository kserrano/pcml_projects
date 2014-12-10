function [ scores ] = predictSVMoutput(Xte, SVMModelTr )

%% predict new data
fprintf('Predict new data...\n');
[~, scores] = predict(SVMModelTr,Xte);

%%
%finalScore = scores(:,1) + scores(:,2);

end

