function [C] = pearsonCorr(tX, y)
% Compute the Pearson correlation in matrix C containing the pairwise 
% linear correlation coefficient between each pair of columns. The last
% column of the matrix C is the correlation between the input and the
% output

C = corr(tX,'type','Pearson');
C = [C corr(tX,y, 'type', 'Pearson')];

end

