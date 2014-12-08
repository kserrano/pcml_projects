function [ prediction ] = kmeansWrapper( data )

[idx,C,sumd,D] = kmeans(data,2);

idx(idx == 2) = 1;
idx(idx == 1) = -1;

prediction = idx;

end

