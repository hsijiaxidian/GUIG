function [SigmaX]=Closed_log(SigmaY,C)
SigmaX = zeros(size(SigmaY));
temp=SigmaY.^2-4*C;
ind=find (temp>0);
SigmaX(ind)=max(SigmaY(ind)+sqrt(temp(ind)),0)/2;

end