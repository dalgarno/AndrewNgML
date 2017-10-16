function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(size(X, 1), 1), X];
total_sum = 0;

Delta_1 = 0;
Delta_2 = 0;

for i=1:m
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %  Forward propagation  %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Select the i'th training example
    x_i = X(i,:);
    
    % Encode the label as one-hot
    y_i = zeros(num_labels, 1);
    y_i(y(i)) = 1;
    
    %Compute the hypothesis for each training example
    % Layer 1
    a_1 = [X(i,:)'];
    
    % Layer 2
    z_2 = Theta1 * a_1;
    a_2 = [1; sigmoid(z_2)];
    
    % Layer 3
    z_3 = Theta2 * a_2;
    h = sigmoid(z_3);
    
    % Compute sum
    inner_sum = (-(y_i)' * log(h)) - ((1 - y_i)' * log(1 - h));
    
    total_sum = total_sum+inner_sum;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %  Backward propagation %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    %remove bias term
    
    del_3 = h - y_i;
    del_2 = Theta2(:,2:end)' * del_3 .* sigmoidGradient(z_2);
    
    Delta_2 = Delta_2 + del_3 * a_2';
    Delta_1 = Delta_1 + del_2 * a_1';
     
% Return cost 
J = total_sum/m;

% Calculate gradients
Theta1_grad = Delta_1./m;
Theta2_grad = Delta_2./m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * Theta2(:,2:end));

% Add regularisation
Theta1_no_bias = Theta1(:, 2:size(Theta1, 2));
Theta2_no_bias = Theta2(:, 2:size(Theta2, 2));
reg = (lambda / (2 * m)) * ...
    (sum(sum(Theta1_no_bias.^2)) + ...
    sum(sum(Theta2_no_bias.^2)));

J = J + reg;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
