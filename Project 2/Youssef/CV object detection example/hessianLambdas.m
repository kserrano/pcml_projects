function [ lambda1, lambda2 ] = hessianLambdas( image )
%computes the hessian of an image and then computes the lambda values for
%each pixel

Ix = imfilter(image, fspecial('prewitt'));
Iy = imfilter(image, (fspecial('prewitt'))');
Ixx = imfilter(Ix, fspecial('prewitt'));
Iyy = imfilter(Iy, (fspecial('prewitt'))');
Ixy = imfilter(Ix, (fspecial('prewitt'))');

% computing the hessian matrix before computing the lambas

lambda1 = 0.5*(Ixx+Iyy - sqrt((Ixx - Iyy).^2 + 4*Ixy.^2 ));
lambda2 = 0.5*(Ixx+Iyy + sqrt((Ixx - Iyy).^2 + 4*Ixy.^2 ));

end

