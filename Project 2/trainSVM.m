function [ SVMModelTr ] = trainSVM( YTr, XTr, kernel, C )
%WRAPPERSVM Summary of this function goes here
%   Detailed explanation goes here

%% create the model
fprintf('Creating the model..\n');
SVMModelTr = fitcsvm(XTr,YTr, 'boxconstraint', C, 'kernel_function', kernel);

end

