%% Load file
close all; clear all;
load('songTrain.mat');
%Ytrain(Ytrain>0) = 1;
ua = Ytrain; % users-artists 
S = artistName;
%% Normalize per user (rows)
rowMax = max(ua,[],2);
%rowMax(rowMax==0) = 1;
%%
rowrep = repmat(rowMax,1,size(ua,2));
ua = ua./rowrep;
%%

RRt = ua*ua';
RtR = ua'*ua;

%% Compute P and Q
[m,n] = size(RRt);
for i = 1:m
   
   diagP(i) = RRt(i,i);
end 
P = diag(diagP);
% for(i = 1:n)
%     col = ua(:,i);
%     diagQ(i) = sum(col);
% end
% Q = diag(diagQ);

%% check if diagQ and diagP has no zero
any(0==diagP)
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
X = Su * ua;
% get the top-k values and their indices
k = 5;
l = 5;
% Get the top k recommendation for the l first users
topKL = cell(l+1,k+1);
for j = 2:k+1
    topKL(1,j)={j-1};
end
for i = 1:l
    [sortX,sortIdx] = sort(X(i,:),'descend'); % sort the scores for user bob
    maxKVal = sortX(1:k)
    maxKindx = sortIdx(1:k);
    topKArtists = S(maxKindx); % get the name of the artists
    topKL(i+1,1) = {i};
    topKL(i+1,2:end) = topKArtists;
end
topKL
%% Are the friends similar?
idx1 = find(Gtrain==1);
simOfFriends = Su(idx1);
meanSimilarity = mean(simOfFriends)

