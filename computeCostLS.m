function L = computeCostLS( y, tX, beta )
%COMPUTECOSTLS compute the cost using MSE

N = length(y);
e = y - tX*beta;
L = e'*e/(2*N);

end

