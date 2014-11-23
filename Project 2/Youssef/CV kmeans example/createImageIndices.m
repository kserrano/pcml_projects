function [ indices ] = createImageIndices(m, n)
% function used to create a matrix containing x, y coordinates in value,
% along each dimension. They go uo to m, and n.

indices(:, :, 1) = repmat((1:m)', 1, n);
indices(:, :, 2) = repmat(1:n, m, 1);

end

