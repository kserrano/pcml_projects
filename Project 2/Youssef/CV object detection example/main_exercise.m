clearvars;
addpath adaboost;

%% Load images and labels
train = [];
test = [];

train.img = im2double(imread('train/img2.png'));
train.labels = im2double(imread('train/img2_labels.png'));


test.img = im2double(imread('test/img1.png'));
test.labels = im2double(imread('test/img1_labels.png'));

% the score image for Exercise 2
test.scoreForEx2 = im2double(imread('test/img1_scores.png'));


%% show one image
figure;

subplot(121);
imshow( train.img );
title('Training Image');

subplot(122);
imshow( train.labels );
title('Training Labels');

%% TASK 1A: Gaussian, Smoothed gradient magnitude & Laplacian of Gaussian

% you have to fill in the code in computeTask1AFeatures.m
% should return a [numel(train.img) 9] feature matrix
task1ATrainFeatures = computeTask1AFeatures( train.img );

% Train classifier
MaxIters = 50;
labels = 2*(train.labels(:) > 0) - 1;
fprintf('Training Classifier..\n');
model = classicABTrain( task1ATrainFeatures, labels, MaxIters );

% Test classifier
fprintf('Testing Classifier..\n');
task1ATestFeatures = computeTask1AFeatures( test.img );
scores = classicABPredict( model, task1ATestFeatures );

figure;
subplot(311); imshow(test.img); title('Original Test image');
subplot(312); imshow(mat2gray(reshape(scores, size(test.img)))); title('TASK 1A predicted image');
subplot(313); imshow( test.labels ); title('Test Labels');

%% Task 1B: Hessian eigenvalues

% you have to fill in the code in computeTask1BFeatures.m
% should return a [numel(train.img) 6 feature matrix
newFeatures = computeTask1BFeatures( train.img );

features = [task1ATrainFeatures, newFeatures];

% re-train with old + new features
MaxIters = 50;
labels = 2*(train.labels(:) > 0) - 1;
fprintf('Training Classifier..\n');
model = classicABTrain( features, labels, MaxIters );

% Test classifier
fprintf('Testing Classifier..\n');
features = [task1ATestFeatures, computeTask1BFeatures( test.img )];
scores = classicABPredict( model, features );

figure;
subplot(311); imshow(test.img); title('Original Test image');
subplot(312); imshow(mat2gray(reshape(scores, size(test.img)))); title('TASK 1B predicted image');
subplot(313); imshow( test.labels ); title('Test Labels');


%% Task 2A: Counting the Number of Synapses

% you have to fill in the code in region_growing.m
% should return a binary mask whose size is the size as the testing image
S_ = test.scoreForEx2;
thresBinary = 170 ./ 255;
B_ = S_ > thresBinary;
T_ = B_ .* S_;
sumThres = 30;
local_peaks = nms(T_, sumThres);

figure; 
showImg =0;
for i = 1:length(local_peaks.J_Mask_Cell)
    tmp_color = rand(1,3);
    tmpJ = local_peaks.J_Mask_Cell{i};
    tmpShowImg = zeros(size(tmpJ,1),size(tmpJ,2),3);
    for k = 1:3
        tmpShowImg(:,:,k) = tmp_color(k) .* tmpJ;
    end
    showImg = showImg + tmpShowImg;
end
imshow(showImg);
tmp_t = sprintf('The # of synapses is %d',length(local_peaks.topN_Score));
title(tmp_t,'FontSize',14);

draw_orie = 0;
if draw_orie
figure; imshow(B_); hold on;
for i = 1:length(local_peaks.J_Mask_Cell)
    tmp1 = local_peaks.J_Mask_Cell{i};
    [FX,FY] = gradient(double(tmp1));
    FXY = gradient(FX);
    FXY_B = FXY > 0;
    FXY_B_Ind = find(FXY_B);
    [I J] = ind2sub(size(FXY),FXY_B_Ind);
    IJ = [I J];
    [V D] = eig(cov(I,J)); V_ = [V(:,1)' * -1; 0 0 ; V(:,1)'];
    plot(local_peaks.topN_Index(i,2),local_peaks.topN_Index(i,1),'bo');
    plot(V_(:,1).*-20+local_peaks.topN_Index(i,2),V_(:,2).*20+local_peaks.topN_Index(i,1), ...
    'Color',rand(1,3),'LineWidth',4);
end
tmp_t = sprintf('Estimated Orientation of the Synapses');
title(tmp_t,'FontSize',14);
end
