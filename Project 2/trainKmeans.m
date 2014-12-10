function [ prediction, centroids, classLabels ] = trainKmeans(y, X )

noOfReplicates = 1;
options = statset('UseParallel', true);

[idx,centroids,~, ~] = kmeans(X,2, 'Options', options, 'Replicates', noOfReplicates);

classes = 1:size(centroids, 1);
prediction = zeros(size(idx));
classLabels = zeros(length(classes), 1);

for i = classes
    
    temp = round(mean(y(idx==i)));
    classLabels(i) = temp;
    prediction(idx == i) = temp;

end

end

