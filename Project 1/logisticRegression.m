function beta = logisticRegression(y,tX,alpha)
% computes the logistic regression beta result using gradient descent.

% 2000 iterations should be enough
maxIters = 5000;

% starting from the beta = D-dimensional zero vector
betaStart = zeros(size(tX, 2), 1);

beta = gradientDescent(y, tX, maxIters, alpha, betaStart,...
    @computeCostLogReg, @computeGradientLogReg, 0, 0);

end

