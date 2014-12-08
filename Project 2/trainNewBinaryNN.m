function [ trainedNN ] = trainNewBinaryNN( dataInput, dataOutput, dimensions, noEpochs,...
    batchSize, plotFlag, learningRate, dropout, lambda, activationFunction )
% function used to set up a new neural network from scratch and train it

assert(dimensions(end) == 2, 'NN can only accept binary outputs');

% prepare labels for NN
extDataOutput = [1*(dataOutput>0)  1*(dataOutput<=0)];  % first column, p(y=1), second column, p(y=-1)

% setup NN. The first layer needs to have number of features neurons
%  and the last layer the number of classes (here two).
trainedNN = nnsetup(dimensions);
opts.numepochs =  noEpochs;   %  Number of full sweeps through data
opts.batchsize = batchSize;  %  Take a mean gradient step over this many samples

if ~isempty(activationFunction)
    trainedNN.activation_function = activationFunction;
end

if ~isempty(dropout)
    trainedNN.dropoutFraction = dropout;
end

if ~isempty(lambda)
    trainedNN.weightPenaltyL2 = lambda;
end

% if == 1 => plots trainin error as the NN is trained
opts.plot = plotFlag; 

trainedNN.learningRate = learningRate;

[trainedNN, ~] = nntrain(trainedNN, dataInput, extDataOutput, opts);

end

