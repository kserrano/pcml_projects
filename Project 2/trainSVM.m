function [ SVMModelTr ] = trainSVM( YTr, XTr, kernel, kernelScale, C )
% use caution with RBF: it completely breaks things !

% create the model
fprintf('Creating the model..\n');

if isempty(C)
    C = 1;
end

if isempty(kernel)
    kernel = 'linear';
end

if isempty(kernelScale)
    kernelScale = 1;
end

SVMModelTr = fitcsvm(XTr,YTr, 'BoxConstraint', C, 'KernelFunction', kernel, 'KernelScale', kernelScale);

end