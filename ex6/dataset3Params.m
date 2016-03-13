function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
s = 5;

test_set_C(1) = 0.01;
test_set_C(2) = 0.03;

test_set_sigma(1) = 0.01;
test_set_sigma(2) = 0.03;

for i = 3:(s-1);
test_set_C(i) = test_set_C(i-2)*10;
test_set_C(i+1) = test_set_C(i-1)*10;
end
test_set_C = test_set_C';
for i = 3:(s-1);
test_set_sigma(i) = test_set_sigma(i-2)*10;
test_set_sigma(i+1) = test_set_sigma(i-1)*10;
end
test_set_sigma = test_set_sigma';

error = zeros(s,s);
for i = 1:s
for j = 1:s
    model = svmTrain(X, y, test_set_C(i), @(x1, x2) gaussianKernel(x1, x2, test_set_sigma(j)));
    prediction = svmPredict(model, Xval);
    error(i,j) = mean(double(prediction ~= yval));
end
end
[I,J] = find(error == min(error(:)));
C = test_set_C(I);
sigma = test_set_sigma(J);


% =========================================================================

end
