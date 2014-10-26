function [ beta ] = gradientDescent(y, tX, maxIters, alpha, betaStart, lossFun, gradFun, lambda)

beta = betaStart;

for k = 1:maxIters
    

    tempBeta = beta;
    tempBeta(1) = 0;
    tempBetaNorm = sqrt(sum((tempBeta).^2));
    
    % computing the loss using the given function, plus norm squared of
    % beta * lambda as a regularization term
    L = lossFun( y, tX, beta ) + lambda*tempBetaNorm^2;
    
    
    % comuting the gradient using the given function, if regularizing then
    % the regularization term's derivative is 2*lambda*beta
    g = gradFun( y, tX, beta ) + 2*lambda*tempBeta;
    
    oldBeta = beta;
    oldBetaNorm = sqrt(sum((oldBeta).^2));
    
    % gradient descen update

    beta = beta - alpha*g;
    newBetaNorm = sqrt(sum((beta).^2));
    
    % tests whether the descent has converged ()
    
    % when the norm of the beta changed by less than 1/1000, time to stop
    if (abs(newBetaNorm-oldBetaNorm)/oldBetaNorm < 10^-3)
        break
    end


    % print current corst and beta
    disp(['cost: ' num2str(L) ' and beta values ' num2str(beta(:)')]);

end

    disp(' ')
    disp(['finished descent after ' num2str(k) ' iterations'])
end

