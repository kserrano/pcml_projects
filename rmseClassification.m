function  error  = rmseClassification( y, tXBeta )
% computes the rmse between a given (categorical) output vector y and the
% probalities obtained from logistic regression, as described on the
% project page

pHat = probEstimate(tXBeta);
error = rmse(y, pHat);

end

