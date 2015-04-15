function [D, V1,V2] = Bhattacharyya_distance(view_margin1, view_margin2)
% Compute the Bhattacharyya_distance between two view classifiers
% y_{i} = sign(w_{i}*x_{i}) = sign(view_margin1(i,:))
% The Bhattacharyya_distance D_{i,j}= -log(sqrt(sigmoid(view_margin1(i))*sigmoid(view_margin2(j)))+...
%  sqrt(1-sigmoid(view_margin1)*(1-sigmoid(view_margin2))))

sigmoid = @(x)(1./(1+exp(-x)));
sigmoid_rev = @(x)(exp(-x)./(1+exp(-x)));

[V1, V2] = meshgrid(view_margin1, view_margin2);

prod_pos = sigmoid(V1).*sigmoid(V2);
prod_neg = sigmoid_rev(V1).*sigmoid_rev(V2);

D = -log(sqrt(prod_pos)+sqrt(prod_neg));





