function [bestParameter, achievedErrorTe ] = analyzeKCVresults( errorTe,...
    errorTr, parameterName, parameterValues )
% plots the results from cross-validating the given parameter, for
% inspecttion and also retrieves the optimal parameter value and returns it

avgErrorTe = mean(errorTe);
avgErrorTr = mean(errorTr);

figure;
plot(parameterValues, errorTe,'r-','color',[1 0.7 0.7]);
hold on
plot(parameterValues, errorTr,'b-','color',[0.7 0.7 1]);
plot(parameterValues, avgErrorTe ,'r-','linewidth', 3); hold on
plot(parameterValues, avgErrorTr,'b-','linewidth', 3); xlabel(parameterName);
ylabel('error');

[achievedErrorTe, bestParameter] = min(avgErrorTe);
bestParameter = parameterValues(bestParameter);

end

