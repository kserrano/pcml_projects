function [ SVMModelTr ] = trainSVM( YTr, XTr, kernel, C )

% create the model
fprintf('Creating the model..\n');

if isempty(C)
    C = 1;
end

if isempty(kernel)
    kernel = 'linear';
end

SVMModelTr = fitcsvm(XTr,YTr, 'BoxConstraint', C, 'KernelFunction', kernel);

end