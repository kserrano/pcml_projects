function loss = loss01( y,haty )
% implements the 0-1 loss error metric given in the project page

N = length(y);
loss = 1/N * sum(y~=haty);

end

