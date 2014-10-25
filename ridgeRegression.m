function beta = ridgeRegression(y,tX,lambda)
% compute beta using the ridgeRegression
[~,M] = size(tX);
beta = (tX'*tX + lambda*eye(M))\tX'*y;

