function g = computeGradientLS( y, tX, beta )
%COMPUTEGRADIANTLS compute the gradient for Least Square
%   
N = length(y);
e = y - tX*beta;
g = -(1/N)*(tX'*e);

end

