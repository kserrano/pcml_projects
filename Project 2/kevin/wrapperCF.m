function rmse = wrapperCF( ua_tr, ua_te, S )
%WRAPPERCF Summary of this function goes here
%   Detailed explanation goes here
% ua_tr_tr : Ytrain_train
% ua_tr_te : Ytrain_test 
% S : Artists_name
%%



%% number of listening in average per user
users_mean = computeMeanPerUser(ua_tr);

%% Normalize per user (rows)
rowMax = max(ua_tr,[],2);
%rowMax(rowMax==0) = 1;
%%
rowrep = repmat(rowMax,1,size(ua_tr,2));
ua_tr = ua_tr./rowrep;
%%
% 
RRt = ua_tr*ua_tr';
RtR = ua_tr'*ua_tr;

%% Compute P and Q
[m,n] = size(RRt);
for i = 1:m
   
   diagP(i) = RRt(i,i);
end 
P = diag(diagP);
% for(i = 1:n)
%     col = ua_tr(:,i);
%     diagQ(i) = sum(col);
% end
% Q = diag(diagQ);

%% check if diagQ and diagP has no zero
any(0==diagP);
%any(0==diagQ)
%% Compute inverse of sqrt P and Q
% Because P,Q are diagonal with non-zero element in the diagonal, the inverse is simply the inverse of the
% diagonal
for i = 1:m
    invSqrtDiagP(i) = sqrt(1/diagP(i));
end
invSqrtP = diag(invSqrtDiagP);

%%
% for i = 1:n
%     invSqrtDiagQ(i) = sqrt(1/diagQ(i));
% end
% invSqrtQ = diag(invSqrtDiagQ);

%% Compute Su and Si as in the pdf: This take a few minutes
Sup1 = bsxfun(@times,diag(invSqrtP),RRt); % optimization for diagonal matrix multiplication (thanks stack overflow)
Su = Sup1*invSqrtP;
%Si = invSqrtQ*RtR*invSqrtQ;
%Sip1 = bsxfun(@times,diag(invSqrtQ),RtR);
%Si = Sip1*invSqrtQ;
fullSu = full(Su);

%% user-user clollaborative (part 3.a)
X = sparse(Su * ua_tr);

%% Get the score for each 0
keyboard
umrep = repmat(users_mean,1,size(X,2));
X = X';
ua_col= X(:);
%idx0 = find(ua_col>0);
%ua_col = ua_col(idx0);
umrep = umrep(:);
%umrep = umrep(idx0);

predictX = ua_col.*umrep;
%for i = 1:size(X0,1)
 %   predictX(i,:) = X0(i,:).*users_mean(i);
%end
%% compute rmse
ua_te = ua_te';
ua_tecol = ua_te(:);
%ua_tecol = ua_tecol(idx0);
keyboard
rmse = computeCost(ua_tecol,predictX)


%% get the top-k values and their indices
k = 5;
l = 5;
% Get the top k recommendation for the l first users
topKL = cell(l+1,k+1);
for j = 2:k+1
    topKL(1,j)={j-1};
end
for i = 1:l
    % check if we remove the non-predicted values
    [sortX,sortIdx] = sort(X(i,:),'descend'); % sort the scores for user bob
    maxKVal = sortX(1:k);
    maxKindx = sortIdx(1:k);
    topKArtists = S(maxKindx); % get the name of the artists
    topKL(i+1,1) = {i};
    topKL(i+1,2:end) = topKArtists;
end


end

