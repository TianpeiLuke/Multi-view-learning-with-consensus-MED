function [D, V1,V2] = exp_distance(view_margin1, view_margin2)
% Compute the Bhattacharyya_distance between two view classifiers
% y_{i} = sign(w_{i}*x_{i}) = sign(view_margin1(i,:))
% The exp D_{i,j}= exp(-sign(f1)*f2-sign(f2)*f1)

[V1, V2] = meshgrid(view_margin1, view_margin2);

f1 = V1;%sigmoid(V1)./sigmoid_rev(V1);
f2 = V2;%sigmoid(V2)./sigmoid_rev(V2);


prod_pos =  sign(f1).*f2;
prod_neg =  sign(f2).*f1;

D = exp(-prod_pos-prod_neg);