%% Load file
close all; clear all;
%load('songTrain.mat');
splitData;
load('songTestPairs');
%Ytrain(Ytrain>0) = 1;
ua_tr = Ytrain_new; % users-artists 
ua_te = Ytest_weak;
ua = Ytrain;
S = artistName;

%%

[predictionWeak,Su_train] = wrapperCF(ua,ua,S);
%%
%[dummy,Su_train] = wrapperCF(ua,ua,S);

%% Are the friends similar?
idx1= find(Gtrain==1);
simOfFriends = Su_train(idx1);
meanSimilarity = mean(simOfFriends)

%%  create the new similarity matrix for new users based on friends
% For position (i,j):  user i will have similarity value mean(Similarity of
% his friends in the train set with user j).
Su_new = zeros(size(Gstrong,1),size(ua,1));

for i = 1:size(Gstrong,1)
    for j = 1:size(ua,1)
        frIdx = find(Gtrain(i,:)==1);
        simOfFriendsForUj = Su_train(frIdx,j);
        Su_new(i,j) = mean(simOfFriendsForUj);
    end
end
Su_new = sparse(Su_new);  
%% create the final similarity matrix to compute the prediction as before
part1 = [Su_train; Su_new];
part2 = [Su_new';eye(size(Su_new,1))];
Su_final = [part1 part2];
%% Compute the strong predictions
% ua_new is the new user-artsts matrix of size 1867x15082, but with 0 only
% in the 93 last rows since we don't have any count for them
ua_new = [ua;zeros(size(Su_new,1),size(ua,2))];
strongX = sparse(Su_final*ua_new);
PredStrong = strongX(end-92:end,:);

%% Store resultsla fprintf('storing results')
fprintf('storing results');
idxw = find(Ytest_weak_pairs>0);
idxs = find(Ytest_strong_pairs>0);
Ytest_weak_pred =Ytest_weak_pairs;
Ytest_strong_pred = Ytest_strong_pairs;
p = predictionWeak(idxw);
p(p==0)=1;
Ytest_weak_pred(idxw) = p;
s = PredStrong(idxs);
s(s==0)=1;
Ytest_strong_pred(idxs) = s;

save('songPred', 'Ytest_strong_pred', 'Ytest_weak_pred');

