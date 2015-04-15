function [D, V1,V2] = quad_distance(view_margin1, view_margin2)
% Compute the Bhattacharyya_distance between two view classifiers
% y_{i} = sign(w_{i}*x_{i}) = sign(view_margin1(i,:))
% The exp D_{i,j}= exp(-sign(f1)*f2-sign(f2)*f1)

[V1, V2] = meshgrid(view_margin1, view_margin2);


S = V1 - V2;
D = S.*S/2;