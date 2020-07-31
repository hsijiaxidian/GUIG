function [ Y,sup,eff_ind ] = str_l0( X,lambda )
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
d = sqrt(sum(X.^2));
eff_ind = 1:size(X,2);
c = d;
mark = d-lambda;
mark_ind = mark<0;
c(mark_ind) = 0;
ind = c==0;eff_ind (ind) = [];
Normili_w = repmat((c./(d+eps)),[size(X,1),1]);
Y= X.*Normili_w;
sup = sum(sign(c));
end

