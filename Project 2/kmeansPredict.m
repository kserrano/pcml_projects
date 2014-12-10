function [classAssignments, distances] = kmeansPredict(data, centroids, classLabels)
% takes clusters trained using kmeans, and predicts labels of new data
% using those

  
  N = size(data, 1);
  
  K = size(centroids, 1);
  
  % for each cluster, find error
  extX = permute(data, [1 3 2]);
  extX = repmat(extX, 1, K, 1); 
  centroids = permute(centroids, [3 1 2]);
  distances = extX - repmat(centroids, N, 1, 1);
  distances = sum(distances.^2, 3);
  distances = distances';
  
  % find assignments
  [distances, classAssignments] = min(distances, [], 1);
  
  for i = 1:K
      classAssignments(classAssignments == i) = classLabels(i);
  end

end

