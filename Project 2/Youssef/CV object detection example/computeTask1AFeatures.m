% You have to fill in the code in this function for Task 1A
function features = computeTask1AFeatures( image )

% features should be a matrix for which the value
%   features(I,J) is the value of the feature J at pixel I

% % This is an example that just places the pixels values in the feature
% % matrix
% % You HAVE TO RE-IMPLEMENT ACCORDING TO THE INSTRUCTIONS
% warning('YOU HAVE TO COMMENT THIS CODE OUT AND PLACE YOURS');
% numPixs = numel(image);
% features = zeros( numPixs, 1 ); % just one feature for the example
% features(:,1) = image(:);   % just the pixel values as the feature

sigmas = [1 2 4];
features = zeros(size(image, 1), size(image, 2), 3*length(sigmas));

for i=1:length(sigmas)
    
    features(:, :, i) = imfilter(image, fspecial('gaussian', sigmas(i)));
    
    features(:, :, i+length(sigmas)) = imageGradient(imfilter...
                    (image, fspecial('gaussian', sigmas(i))));
    
    features(:, :, i+2*length(sigmas)) = imfilter(image, fspecial('log', sigmas(i)));
    
end

features = reshape(features, [], size(features, 3));

end
