function [ beta ] = leastSquares( y, Tx )
% direct LS computation method, using the fomula from the lectures

beta = (Tx'*Tx)\(Tx'*y);

end

