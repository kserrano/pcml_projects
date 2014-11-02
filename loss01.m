function loss = loss01( y,haty )
%LOSS01 computes the 0-1 loss 

N = length(y);
loss = 1/N * sum(y~=haty);

end

