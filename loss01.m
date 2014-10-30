function loss = loss01( y,haty )
%LOSS01 Summary of this function goes here
%   Detailed explanation goes here
N = length(y);
loss = 1/N * sum(y~=haty);

end

