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

X = [ones(m, 1) X];

% size(X) % 5000 X 401
% size(y) % 5000 X 1
%size(Theta1) % 25 X 401
%size(Theta2) % 10 X 26

for i = 1:m

    xOne = X(i,:);

    %a2 = sigmoid(Theta1*(xOne'));
    a2 = sigmoid(xOne*(Theta1'));
    a2 = [ones(1, 1) a2];

    %size((-y.*(log(sigmoid(a2*(Theta2')))))-((1-y).*(log(1-sigmoid(a2*(Theta2')))))) % 5000 X 10
    %size(log(sigmoid(a2*(Theta2')))) %  1 X 10
    %size(y) % 5000 * 1

    if (size(Theta2, 1) == 10)
        yi = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 9)
        yi = [0; 0; 0; 0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 8)
        yi = [0; 0; 0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 7)
        yi = [0; 0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 6)
        yi = [0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 5)
        yi = [0; 0; 0; 0; 0];    
    elseif (size(Theta2, 1) == 4)
        yi = [0; 0; 0; 0];                               
    else
        yi = [0; 0; 0];    
    endif
    
    %fprintf('\n Yi = %d',y(i,:));
    yi(y(i,:)) = 1;
    %yi
    %(-yi.*(log(sigmoid(a2*(Theta2')))))
    J += sum((-yi.*(log(sigmoid(a2*(Theta2'))))')-((1-yi).*(log(1-sigmoid(a2*(Theta2'))))'));
endfor

J = ((1/m)*J) + ((lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2), 2) + sum(sum(Theta2(:,2:end).^2), 2)));

%fprintf('\n J = %d', J);
bigDel1 = 0;
bigDel2 = 0;

%size(Theta1) % 5*4
%size(Theta2) % 3*6

for i = 1:m

    xOne = X(i,:);

    z2 = (Theta1 * xOne');

    %size(z2) %5*1
    
    z2 = [ones(1, 1); z2];
    a2 = sigmoid(Theta1 * xOne');
    a2 = [ones(1, 1); a2];
    %size(a2) %6*1

    Z3 = Theta2*a2;
    a3 = sigmoid(Z3);
    %size(a3) %3*1

    if (size(Theta2, 1) == 10)
        yi = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 9)
        yi = [0; 0; 0; 0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 8)
        yi = [0; 0; 0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 7)
        yi = [0; 0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 6)
        yi = [0; 0; 0; 0; 0; 0];
    elseif (size(Theta2, 1) == 5)
        yi = [0; 0; 0; 0; 0];    
    elseif (size(Theta2, 1) == 4)
        yi = [0; 0; 0; 0];                               
    else
        yi = [0; 0; 0];    
    endif
    
    yi(y(i,:)) = 1;

    del3 = a3 - yi;
    %size((Theta2)'*del3) % 6*1
    %size(sigmoidGradient(z2)) % 6*1

    dpart1 = (Theta2)'*del3;
    dpart2 = sigmoidGradient(z2);

    del2 = dpart1.*dpart2; % (6*1).*(6*1)
    %size(del2) % 6*1

    del2 = del2(2:end);
    %size(del2) % 5*1
    
    %size(del3) %3*1
    %size(sigmoidGradient(z2))

    %size(del3*(a2)')
    %size(bigDel2 + ((del3*(a2)')))

    bigDel1 = bigDel1 + (del2*xOne);
    bigDel2 = bigDel2 + (del3*(a2)');

endfor

Theta1_grad = (1/m)*bigDel1;
Theta2_grad = (1/m)*bigDel2;

Theta1_NoBias = Theta1(:, 2:end);
Theta2_NoBias = Theta2(:, 2:end);

Theta1_NoBias = [zeros(size(Theta1, 1), 1), Theta1_NoBias];
Theta2_NoBias = [zeros(size(Theta2, 1), 1), Theta2_NoBias];

Theta1_grad += ((1/m)*lambda)*Theta1_NoBias;
Theta2_grad += ((1/m)*lambda)*Theta2_NoBias;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%size(Theta1_grad)
%size(Theta2_grad)


end
