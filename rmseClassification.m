function  error  = rmseClassification( y, tXBeta )

pHat = probEstimate(tXBeta);
error = rmse(y, pHat);

end

