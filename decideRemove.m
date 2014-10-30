function rA = decideRemove( A )
% Decide which variables to remove from the Dx(D+1) matrix A.
% We can remove one input variable v if the correlation between the v and
% the ouput is zero and that all the row of v in A is 0 (except v corr v)
outputCulumn = size(A,2);
maxCorr = max(A(:,outputCulumn));
threshold = 0.10*maxCorr;
for i = 1:size(A,1)
   if(A(i,outputCulumn)< threshold) % Ouput corr is less thant the threshold
       % check if each element of this row is < threshold
       s = sum(A(i,:)< threshold);
       if(s==(size(A,1)-1)) % if all element of the row (except in diagonal) are less than the threshold then remove variable
           A(i,:) = [];
           A(:,i) = [];
       end
   end
end
rA = A;
end

