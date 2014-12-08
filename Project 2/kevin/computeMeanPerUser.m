function [ m ] = computeMeanPerUser( ua )
%% Take the mean per user
% take the sum of listening for each user and take the mean 
nzeros_row = sum(ua~=0,2);
ua = ua';
rowsum = sum(ua);

m = rowsum'./nzeros_row;
end

