function [ result ] = rmse( x, y )
% computes the RMSE between two data vectors

result = sqrt(nanmean((x(:)-y(:)).^2));

end

