function [ beta ] = leastSquares( y, Tx )
% Compute beta using least squares

beta = (Tx'*Tx)\(Tx'*y);

end

