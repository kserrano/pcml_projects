function [ trainedNN ] = trainNewNN( dataInput, dataOutput, dimensions, noEpochs, batchSize, plotFlag, learningRate )
% function used to set up a new neural network from scratch and train it

% setup NN. The first layer needs to have number of features neurons
%  and the last layer the number of classes (here two).
trainedNN = nnsetup(dimensions);
opts.numepochs =  noEpochs;   %  Number of full sweeps through data
opts.batchsize = batchSize;  %  Take a mean gradient step over this many samples

% if == 1 => plots trainin error as the NN is trained
opts.plot = plotFlag; 

trainedNN.learningRate = learningRate;

[trainedNN, ~] = nntrain(trainedNN, dataInput, dataOutput, opts);

end

