function [labelImage] = clusterImage(depthImage)
%
% Inputs: 
% 
% depthImage: An image where each value indicates the depth of the
% corresponding pixel.
%
% Outputs: 
%
% labelImage:  Output label image where each value indicates to which 
% cluster the corresponding pixels belongs. There are three clusters: 
% value 1 for the background, value 2 for the hand and value 3 for 
% the doll.
%

moveThreshold = 1/100;
maxIterations = 1000;

k=3;
% initiate centroids - BEGIN
% we create 3 centroids, in the format [Z X Y], where Z in the inverse depth
mu=zeros(k,3);
mu(1,:) = [0 0 0]; %background centroid
mu(2,:) = [500 0 0]; %hand centroid
mu(3,:) = [1000 0 0]; %doll centroid
% initiate centroids - END

% depthImage = depthImage*1000;

[m, n] = size(depthImage);

coordinates(:, :, 1) = depthImage;
coordinates(:, :, 2:3) = createImageIndices(m, n);

noIterations = 1;

move = moveThreshold + 1;

labelImage = uint8(size(depthImage));

while move > moveThreshold && noIterations <= maxIterations

    distances(:, :, 1) = sum(abs(coordinates - repmat(reshape(mu(1, :)...
                                , 1, 1, []), m, n)).^2, 3).^0.5;
    distances(:, :, 2) = sum(abs(coordinates - repmat(reshape(mu(2, :)...
                                , 1, 1, []), m, n)).^2, 3).^0.5;
    distances(:, :, 3) = sum(abs(coordinates - repmat(reshape(mu(3, :)...
                                , 1, 1, []), m, n)).^2, 3).^0.5;
    [~, labelImage] = min(distances, [], 3);

    cluster1 = find(labelImage == 1);
    cluster2 = find(labelImage == 2);
    cluster3 = find(labelImage == 3);

    coordinates = reshape(coordinates, [], 3);

    oldMus = mu;

    mu(1,:) = [mean(coordinates(cluster1, 1)) mean(coordinates(cluster1...
                    , 2)) mean(coordinates(cluster1, 3))];
    mu(2,:) = [mean(coordinates(cluster2, 1)) mean(coordinates(cluster2...
                    , 2)) mean(coordinates(cluster2, 3))];
    mu(3,:) = [mean(coordinates(cluster3, 1)) mean(coordinates(cluster3...
                    , 2)) mean(coordinates(cluster3, 3))];

    coordinates = reshape(coordinates, m, n, 3);  
    move = max(sum( abs((oldMus - mu)./oldMus).^2,  1));

    noIterations = noIterations + 1;

end
            
end