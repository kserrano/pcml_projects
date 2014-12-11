function [ finalScore ] = predictSVMoutput(Xte, SVMModelTr )

% predict new data
fprintf('Predict new data...\n');
[~, scores] = predict(SVMModelTr,Xte);

% Assuming -1 and +1 labels, as is the case for our project
% finalScore = scores(:, 2)-min(scores(:, 2));
% finalScore = finalScore./max(finalScore);
% finalScore = 2*finalScore-1;

finalScore = scores(:, 2)

end

