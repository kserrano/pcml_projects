function [ beta ] = leastSquares( y, Tx )

beta = (Tx'*Tx)\(Tx'*y);

end

