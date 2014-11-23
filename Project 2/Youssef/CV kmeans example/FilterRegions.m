function [FilteredImage] = FilterRegions( OrigImage, BinaryForegroundMap, ...
                                          SharpeningAlpha, SmoothingSigma  )
%
% Inputs: 
% 
% OrigImage: RGB or grayscale input image to be filtered.
%
% BinaryForegroundMap: Binary map of the foreground region(s). The type 
% (class) of the input image must be logical.
%
% SharpeningAlpha: The parameter of the sharpening filter.
% 
% SmoothingSigma: Standard deviation of the gaussian filter.
%
% Outputs: 
%
% FilteredImage:  2D filtered image that has the same type and dimension 
% as the original one.
%

delta = [0 0 0; 0 1 0; 0 0 0];

sharpeningFilter = delta - fspecial('laplacian', SharpeningAlpha);
gaussianFilter = fspecial('gaussian', 7, SmoothingSigma);

OrigImage = im2double(OrigImage);

background = imfilter(OrigImage, gaussianFilter);
foreground = imfilter(OrigImage, sharpeningFilter);

FilteredImage = zeros(size(OrigImage));

FilteredImage(BinaryForegroundMap) = foreground(BinaryForegroundMap);
FilteredImage(~BinaryForegroundMap) = background(~BinaryForegroundMap);

end