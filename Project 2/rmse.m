function [ result ] = rmse( x, y )
% computes the RMSE between two data vectors

result = sqrt(mean((x(:)-y(:)).^2));

end

