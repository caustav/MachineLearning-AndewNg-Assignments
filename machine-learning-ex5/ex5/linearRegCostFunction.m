function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%size(grad)


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

thetaReg = theta;
thetaReg(1) = 0;

J = (1/(2*m))*sum(((X*theta) - y).^2) + (lambda/(2*m))*sum(thetaReg.^2);

for i = 1:size(theta, 1)
    g = (1/m)*sum(((X*theta) - y).*X(:, i));
    if (i > 1)
        g += (lambda/m)*theta(i);
    endif
    grad(i) = g;
endfor

% =========================================================================
%size(grad)
grad = grad(:);

end
