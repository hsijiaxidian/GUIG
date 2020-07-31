function [ Y,sup,eff_ind] = str_lp( X,lambda,p )
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
d = sqrt(sum(X.^2));
% drops = d(1:end-1)./d(2:end);[dmx,imx] = max(drops);
% d(imx+1:end) = 0;
eff_ind = 1:size(X,2);
if p==0.5
    c = max(d-0.5*lambda,0);
    ind = c==0;eff_ind (ind) = [];
    Normili_w = repmat((c./(d+eps)),[size(X,1),1]);
    Y= X.*Normili_w;
elseif p==1
    c = d/(1+lambda);
    Y = X/(1+lambda);
elseif p<0.5
    c = solve_Lp(d,0.5*lambda,2*p);
    ind = find(c==0);
    Normili_w = repmat((c./(d+eps)),[size(X,1),1]);
    Y = X.*Normili_w;
else
    error('The specific "p" is not supported right now :) \n')
end
sup = sum(c.^(2*p));

end

function   x   =  solve_Lp( y, lambda, p )
   if p==1
       J =   1;
   elseif p<1;
       J =   5;
   end
    tau   =  (2*lambda.*(1-p))^(1/(2-p)) + p*lambda.*(2*(1-p)*lambda)^((p-1)/(2-p));
    x     =   zeros( size(y) );
    i0    =   find( y>tau );

    if length(i0)>=1
        % lambda  =   lambda(i0);
        y0    =   y(i0);
        t     =   y0;
        for  j  =  1 : J
            t    =  y0 - p*lambda.*(t).^(p-1);
        end
        x(i0)   =  max(t,0);
    end

    % f=@(x,y)lambda*x^p+1/2*(x-y)^2;
    % error=zeros(1,length(i0));
    % for j=1:length(i0)
    %   error(j)=f(0,y(i0(j)))-f(x(i0(j)),y(i0(j)));
    % end
    % error
end