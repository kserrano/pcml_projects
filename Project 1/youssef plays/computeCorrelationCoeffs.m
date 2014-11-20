function [ spearmanCoeffs, spearmanSig, pearsonCoeffs, pearsonSig,...
    normalityTests, corrExistence, corrSuitability] = ...
    computeCorrelationCoeffs( statistics, noOfVariables )
% computes the correlation coefficient (both pearson & spearman) for each
% input variable with respect to output variable

% initialization of the normality test results
normalityTests = zeros(noOfVariables, 2);


% normality testing for each variable (columns) after pooling.
[normalityTests(:, 1), normalityTests(:, 2)] =...
    checkNormality( statistics, noOfVariables );


% computation of IQM's spearman and pearson coefficients, using each
% success criterion seperately
[spearmanCoeffs, spearmanSig] = corr( statistics(:, 1), statistics(:, 2:end)...
    ,'type', 'spearman', 'rows', 'complete');
[pearsonCoeffs, pearsonSig] = corr( statistics(:, 1),...
    statistics(:, 2:end), 'rows', 'complete');

% deriving the hypothesis test results, 1 = relationship exists, 0 not
spearmanSig = spearmanSig < 0.05;
pearsonSig = pearsonSig < 0.05;

% computation of whether the IQMs, under each scenario, have ANY
% statistical relationship with the succes criteria
corrExistence = spearmanSig | pearsonSig;

% computation of which correlation method best suits each IQM under a
% specific success criterion (useful only for simple scenario).
corrSuitability = pearsonSig - spearmanSig;
% - 1 means spearman preferred, 0 neither, +1 pearson preferred

end