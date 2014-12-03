function L= computeCostMean(y,m)
N = length(y);
e = y - m;
[n1 n2 ] = size(e);
L = (sum(e(:).^2))/(n1*n2);
end