% An example of matrix completion with noise using PALM
%% Generate problem data
% clc;clear all; close all
% warning off
addpath(genpath(pwd))
rand('seed', 0); randn('seed', 0);
m = 2000; n = 2000; % matrix dimension
r = 5; % matrix rank
L = randn(m,r); U = randn(r,n); % nonnegative factors
Mtrue = L*U; % true matrix
% -- add noise --
N = randn(m,n);
sigma = 0.1;
M = Mtrue + sigma*N;
Ground = M;
sr = 0.035; % percentage of samples
known = randsample(m*n,round(sr*m*n)); % randomly choose samples
data = M(known);
[known,Id] = sort(known); data = data(Id); % observated entries
Input = zeros(m,n); Input(known) = data;
support = zeros(m,n); support(known) = 1;
%% Solve problem
p_i=[0.5,0.5];        %1/p=  \sum_{i=1}^{I} 1/p_i
k=round(r*1.25);                 %Estimated Rank
% k = m;
for i=1:length(p_i)
     fprintf('p_%1d=%1d, ',i,p_i(i))
end
fprintf('\n')
Inputsp = sparse(Input);
R = randn(n,k);
t0 = tic;
U = powerMethod(Inputsp,R,3,1e-6);
[r,s,v] = svd(U'*Inputsp,'econ');u = U*r;
% [u,s,v] = lansvd(Inputsp,k); % 尝试用Power method给一个初始值，不用svd或lansvd
S = diag(diag(s).^(1/2));
X = u(:,1:k)*S(1:k,1:k);
Y = S(1:k,1:k)*v(:,1:k)';
% X = randn(m,k);Y = randn(k,n);

clear opts
opts.X = X;
opts.Y = Y;

opts.groundtruth = Mtrue;
opts.show_progress = false;    % Whether show the progress  
opts.show_interval = 100;   
opts.eta = 0.05;               %  sr 越大 eta 越大； n=1000时 eta=0.4左右(n越大eta越小) n<1000, eta>0.1 应该是X'X越大eta越小，X'X越小eta越大
opts.lambda = 1*norm(data,2);            % 1 for lp
opts.mu = 1e-4*opts.lambda;
opts.p_i = p_i;
opts.p = p_i(1);
opts.maxit = 500;            % Max Iteration  
opts.tol = 1e-3;        
opts.maxtime = 8e3;           % Sometimes terminating early is good for testing. 
opts.gate_upt = 1/2;          % The gate indicating which factor to be updated
siz.m = m; siz.n = n;  siz.k = k;
[X,Y, Out] = Smf_lr_PALM(Input,support,siz,opts);
Result = X*Y;
time = toc(t0);
rank(Result)

%% Reporting

fprintf('time = %4.2e,\n ', time);
fprintf('solution relative error = %4.2e\n\n', norm(Result - Mtrue,'fro')/norm(Result,'fro'));


