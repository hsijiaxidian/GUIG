function [ Y,sup, eff_ind ] = str_log( X,lambda )
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
eff_ind = 1:size(X,2);
d = sqrt(sum(X.^2));
c = Closed_log(d,lambda);
ind = c==0;eff_ind (ind) = [];
Normili_w = repmat((c./(d+eps)),[size(X,1),1]);
Y = X.*Normili_w;
sup = sum(log(c + eps));
end
