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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Cost
h = X * theta;
cost_first_sum = (h - y)' * (h - y);

% Don't want to regularise bias
theta(1) = 0;
cost_second_sum = theta' * theta;

J = cost_first_sum/(2 * m) + (cost_second_sum * lambda)/(2 * m);

% Gradient
grad = (X' * (h - y))/m + (lambda/m)*theta;










% =========================================================================

grad = grad(:);

end
