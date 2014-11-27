%% Load file
users_shows = fopen('user-shows.txt','r');
sizeR= [563 9984];
formatSpec = '%f';
R = fscanf(users_shows,formatSpec,sizeR)';
TS = textread('shows.txt','%q');
%% 100 TV shows and Bob
S= TS(1:100);

bob = R(200,:);
%% Test case (from my pdf, used for debug my code)
testR = [0 1 0 1 1 ; 1 0 0 1 0; 0 1 1 1 0];
testP = [3 0 0 ; 0 2 0 ; 0 0 3];
testQ = [1 0 0 0 0; 0 2 0 0 0 ; 0 0 1 0 0; 0 0 0 3 0;0 0 0 0 1];

%% Compute P and Q
[m,n] = size(R);
for i = 1:m
    row = R(i,:);
   diagP(i) = sum(row);
end 
P = diag(diagP);
for(i = 1:n)
    col = R(:,i);
    diagQ(i) = sum(col);
end
Q = diag(diagQ);

%% check if diagQ and diagP has no zero
any(0==diagP)
any(0==diagQ)
%% Compute inverse of sqrt P and Q
% Because P,Q are diagonal with non-zero element in the diagonal, the inverse is simply the inverse of the
% diagonal
for i = 1:m
    invSqrtDiagP(i) = sqrt(1/diagP(i));
end
invSqrtP = diag(invSqrtDiagP);

%%
for i = 1:n
    invSqrtDiagQ(i) = sqrt(1/diagQ(i));
end
invSqrtQ = diag(invSqrtDiagQ);
%%

RRt = R*R';
RtR = R'*R;
%% Compute Su and Si as in the pdf: This take a few minutes
Sup1 = bsxfun(@times,diag(invSqrtP),RRt); % optimization for diagonal matrix multiplication (thanks stack overflow)
Su = Sup1*invSqrtP;
%Si = invSqrtQ*RtR*invSqrtQ;
Sip1 = bsxfun(@times,diag(invSqrtQ),RtR);
Si = Sip1*invSqrtQ;

%% user-user clollaborative (part 3.a)
X = Su * R;
% get the top-k values and their indices
k = 5;
[sortX,sortIdx] = sort(X(200,1:100),'descend'); % sort the first 100 shows scores for user bob
maxKVal = sortX(1:k)
maxKindx = sortIdx(1:k);
topKshows = S(maxKindx) % get the name of the shows
%% movie-movie collaborative (part 3.b)
Y = R * Si;
k = 5;
% get the top-k values and their indices
[sortY,sortIdy] = sort(Y(200,1:100),'descend');
maxKValy = sortY(1:k)
maxKindy = sortIdy(1:k);
topKshowy = S(maxKindy) % get the name of the shows
%% Part 3.c
bobFile = fopen('bob.txt','r');
sizeBob= [563 1];
formatSpec = '%f';
bobM = fscanf(bobFile,formatSpec,sizeBob)';
%user-user filtering
watched = 0;
for i = 1:k
    if(bobM(sortIdx(i))==1)
        watched = watched+1;
    end
end
prec1 = watched/k
%item-item filtering
watched = 0;
for i = 1:k
    if(bobM(sortIdy(i))==1)
        watched = watched+1;
    end
end
prec2 = watched/k
