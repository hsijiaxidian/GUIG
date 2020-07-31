
addpath('data')
addpath('Utilities')  % It is better to mex the files in this fold first
dataset = 4;
namelist = {'moive-100K','moive-1M','moive-10M','Netflix'};
%You can download the full dataset from https://drive.google.com/open?id=0B244f561lTfqazF4U0MyZjNfc2c
fname = namelist{dataset};
load(fname,'M');
if size(M,1) > size(M,2)
    M = M';
end
normB = sqrt(sum(M.*M));
zerocolidx = find(normB==0);
if ~isempty(zerocolidx)
    fullcol = setdiff(1:size(M,2), zerocolidx);
    M = M(:,fullcol);
end
clear fullcol

[m,n] = size(M);
[I_all,J_all,data_all] = find(M);
fprintf('Size is %d * %d, Sample Rate is %g \n', m, n, length(I_all)/m/n)
clear M

% data_all = data_all - mean(data_all);
% data_all = data_all/std(data_all);
rng(1,'twister');

train_rate = 0.80;  
trian_sample = randperm(length(I_all),round(train_rate*length(I_all)));
trian_sample = sort(trian_sample);
test_sample = setdiff(1:length(I_all), trian_sample);
I_train = I_all(trian_sample);       I_test = I_all(test_sample);             clear I_all 
J_train = J_all(trian_sample);       J_test = J_all(test_sample);             clear J_all 
data_train = data_all(trian_sample); data_test = data_all(test_sample);       clear data_all
clear trian_sample test_sample
L=length(I_train);

support=sparse(I_train,J_train,1,m,n);
Input=sparse(I_train,J_train,data_train,m,n);

p_i=[0.5,0.5];        %1/p=  \sum_{i=1}^{I} 1/p_i
k=15;                 %Estimated Rank
for i=1:length(p_i)
     fprintf('p_%1d=%1d, ',i,p_i(i))
end
fprintf('\n')

%% Initialization
R = randn(n,k);
t0 = tic;
U = powerMethod(Input,R,3,1e-6);
[r,s,v] = svd(U'*Input,'econ');
u = U*r;
% [u,s,v] = lansvd(Input,k); % 尝试用Power method给一个初始值，不用svd或lansvd
S = diag(diag(s).^(1/2));
X = u(:,1:k)*S(1:k,1:k);
Y = S(1:k,1:k)*v(:,1:k)';

clear opts
opts.X = X;
opts.Y = Y;
opts.I_test = I_test;opts.J_test = J_test;
opts.data_test = data_test;

opts.t0 = t0;
opts.show_progress = false;    % Whether show the progress  
opts.show_interval = 100;   
opts.eta = 0.25;               %  rate = 0.5 eta = 0.2， mu = 0.3; rate = 0.7, eta = 0.3, mu = 0.35; rate = 0.8, eta = 0.06; For dataset = 2;
opts.lambda = 2*norm(data_train,2);            % 1 for lp
opts.mu = 0.05*opts.lambda;
opts.p_i = p_i;
opts.p = p_i(1);
opts.maxit = 150;            % Max Iteration   300
opts.tol = 1e-3;        
opts.maxtime = 8e3;           % Sometimes terminating early is good for testing. 
opts.gate_upt = 1/2;          % The gate indicating which factor to be updated
siz.m = m; siz.n = n;  siz.k = k;
[X,Y,out] = Smf_lr_PALM(Input,support,siz,opts);
% RMSE = CompRMSEm(X*Y,opts.I_test+ m*(opts.J_test-1), opts.data_test);
time = toc(t0);
rate=num2str(train_rate);
record_name = ['Time_RMSE_GUIG_log_',rate,'_',fname,'.mat'];
save(['D:\贾西西\2017工作\非凸低秩矩阵分解方法\低秩矩阵分解代码\GUIG_real_MC\Result_record\',record_name],'out')
% rank(X*Y)

%% Reporting

fprintf('time = %4.2e,\n ', time);
% fprintf('solution relative error = %4.2e\n\n', RMSE);
