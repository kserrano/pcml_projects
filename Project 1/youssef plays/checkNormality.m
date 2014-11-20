function [ normalityTests, pValues ] = checkNormality( results, noOfVariables )
% testing the normal distribution hypothesis for each IQM (column), and also
% for the success criteria (also used as variable in computing coefficients
% in other functions) of the results, seperately for each pool (scenario,
% real-world photo). Test based on the One-sample Kolmogorov-Smirnov method

% initialization
normalityTests = zeros(noOfVariables, 1);
pValues = zeros(noOfVariables, 1);

% normality tests computations for the IQMs
for k = 1:noOfVariables
    temp = results(:, k);
    
    if sum(isnan(temp)) == size(temp, 1)
        normalityTests(k) = 0;
        pValues(k) = NaN;
    else
        
    [normalityTests(k), pValues(k)] = kstest(temp);
    end
end

end