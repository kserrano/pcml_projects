function beta = ridgeRegression(y,tX,lambda)
% compute beta using the ridgeRegression, using the direct formula form the
% lectures

[~,M] = size(tX);
beta = (tX'*tX + lambda*eye(M))\tX'*y;

