function [ beta ] = gradientDescent(y, tX, maxIters, alpha, betaStart, lossFun, gradFun, lambda)

beta = betaStart;

minIters = maxIters/5;

for k = 1:maxIters
    

    tempBeta = beta;
    tempBeta(1) = 0;
    tempBetaNorm = sqrt(sum((tempBeta).^2));
    
    % computing the loss using the given function, plus norm squared of
    % beta * lambda as a regularization term
    L = lossFun( y, tX, beta ) + 1/2*1/length(y)*lambda*tempBetaNorm^2;
    
    
    % comuting the gradient using the given function, if regularizing then
    % the regularization term's derivative is 2*lambda*beta
    g = gradFun( y, tX, beta ) + 1/length(y)*lambda*tempBeta;
    
    oldBeta = beta;
    
    % gradient descen update
    beta = beta - alpha*g;
    
    % tests whether the descent has converged ()
    
    % when the norm of the beta changed by less than 1/1000, time to stop
    if (min(abs(beta-oldBeta)./oldBeta) < 10^-3) && (k > minIters)
        break
    end

    % print current corst and beta
    disp(['cost: ' num2str(L) ' and beta values ' num2str(beta(:)')]);

end

    disp(' ')
    disp(['finished descent after ' num2str(k) ' iterations'])
end

