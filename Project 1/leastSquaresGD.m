function beta = leastSquaresGD(y,tX,alpha)
% compute beta using the gradient descent with least squares

maxIters = 5000;
lambda = 0;
betaStart = zeros(size(tX,2),1);

beta = gradientDescent(y,tX, maxIters, alpha, betaStart, @computeCostLS, ...
    @computeGradientLS,lambda, 0);

end

