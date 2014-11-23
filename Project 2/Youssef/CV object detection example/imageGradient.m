function [ result ] = imageGradient( image )
% computes the gradient of an image

derivativeHor = imfilter(image, fspecial('prewitt'));
derivativeVer = imfilter(image, (fspecial('prewitt'))');

result = (derivativeHor.^2 + derivativeVer.^2).^(0.5);

end

