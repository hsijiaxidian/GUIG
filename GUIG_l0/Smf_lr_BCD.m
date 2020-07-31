function [X,Y,Out] = Smf_lr_BCD(data,known,m,n,r,opts)
%  nmfc: nonnegative matrix factorization from partial observations by block-coordinate update
%   minimize 0.5*||P_known(M - X*Y)||_F^2
%   subject to X>=0, Y>=0
%
%  input:
%       data: observed entries of a nonnegative matrix
%       known: index set of observed entries
%       r: estimated rank (X has r columns and Y has r rows)
%       opts.
%           tol: tolerance for relative change of function value, default: 1e-4
%           maxit: max number of iterations, default: 500
%           maxT: max running time, default: 1e3
%           rw: control the extrapolation weight, default: 1
%           X0, Y0: initial X and Y, default: Gaussian random matrices
%  output:
%       X, Y: solution nonnegative factors
%       out.
%           iter: number of iterations
%           hist_obj: objective value at each iteration
%           hist_rel: relative changes at each iteration
%
% More information can be found at:
% http://www.caam.rice.edu/~optimization/bcu/

%% Parameters and defaults
if isfield(opts,'tol'),    tol = opts.tol;     else tol = 1e-4;   end % stopping tolerance
if isfield(opts,'maxit'),  maxit = opts.maxit; else maxit = 500;  end % max # of iterations
if isfield(opts,'maxT'),   maxT = opts.maxT;   else maxT = 1e3;   end % max time in seconds
if isfield(opts,'rw'),     rw = opts.rw;       else rw = 1;       end % initial extrapolation weight
if isfield(opts,'lambda'),     lambda = opts.lambda;       else rw = 1;       end 
if isfield(opts,'eta'),     eta = opts.eta;    else eta = 1;      end 
if isfield(opts,'mu'),      mu = opts.mu;      else mu = 1;       end 
if isfield(opts,'p'),       p  = opts.p;      else p = 0.5;       end 
%% Data preprocessing and initialization
if isfield(opts,'X0'), X0 = opts.X; else X0 = max(0,randn(m,r)); end % initial X
if isfield(opts,'Y0'), Y0 = opts.Y; else Y0 = max(0,randn(r,n)); end % initial Y
[known,Id] = sort(known); data = data(Id); % observated entries

M = zeros(m,n); M(known) = data; % initial M is 0-filled
M0 = M;
nrmb = norm(data); % norm of data
Mnrm = nrmb; % norm of M

% X0 = X0/norm(X0,'fro')*sqrt(nrmb); % normlize X
% Y0 = Y0/norm(Y0,'fro')*sqrt(nrmb); % normlize Y
Xm = X0; Ym = Y0; % extrapolations of X, Y
Yt = Y0'; Ys = Y0*Yt; MYt = M*Yt; % cache useful computation
fsp = @(X,p) sum((sum(X.^2)).^p);
sxp = fsp(X0,p);syp = fsp(Y0',p);
obj0 = 0.5*Mnrm^2 + 0.5*lambda*sxp + 0.5*lambda*syp; % initial objective

t0 = 1; % used for extrapolation weight update
Lx = 1; Ly = 1; % initial Lipschitz constants for X/Y-updates

%% Iterations of block-coordinate update 
%
%  iteratively updated variables:
%       M: estimated matrix. M(known) is fixed; M(~known) is iterative updated
%       Gx, Gy: gradients with respect to X, Y
%       X, Y: new updates
%       Xm, Ym: extrapolations of X, Y
%       Lx, Lx0: current and previous Lipschitz bounds used in X-update
%       Ly, Ly0: current and previous Lipschitz bounds used in Y-update
%       obj, obj0: current and previous objective values
fprintf('Iteration:     ');
start_time = tic;

for k = 1:maxit
    fprintf('\b\b\b\b\b%5i',k); 
    lambda = max(0.85*lambda,mu);
    % --- X-update ---
    Lx0 = Lx;    Lx = eta*norm(Ys,'fro'); % save and update Lipschitz bound for X
    Gx = Xm*Ys - MYt; % gradient at X=Xm
    [X,sxp]= str_lp(Xm - Gx/Lx,lambda/Lx,p);
    Xt = X'; Xs = Xt*X; % cache useful computation
    
    % --- Y-update ---
    Ly0 = Ly; Ly = eta*norm(Xs,'fro');  % save and update Lipschitz bound for Y
    Gy = Xs*Ym - Xt*M; % gradient at Y=Ym
    [Y,syp]=str_lp((Ym - Gy/Ly)',lambda/Ly,p);Y = Y';
    Yt = Y'; Ys = Y*Yt; % cache useful computation
    
    % --- M-update ---    
    M = X*Y;    M(known) = data;         
    Mnrm = norm(M,'fro');        
    MYt = M*Yt; % cache useful computation
    
    % --- diagnostics, reporting, stopping checks ---
    obj = 0.5*(sum(sum(Xs.*Ys))-2*sum(sum(X.*MYt))+Mnrm^2);
    relerr1 = abs(obj-obj0)/(obj0+1);    relerr2 = sqrt(2*obj)/nrmb;
    
    % reporting
    Out.hist_obj(k) = obj;
    Out.hist_rel(1,k) = relerr1; 
    Out.hist_rel(2,k) = relerr2;
    
    % check stopping criterion   
    crit = relerr1<tol; 
    if crit; nstall = nstall+1; else nstall = 0; end
    if nstall>=3 || relerr2<tol; break; end
    if toc(start_time) > maxT; break; end;   
    
    % --- correction and extrapolation ---
    t = (1+sqrt(1+4*t0^2))/2;    
    if obj>obj0 
        % restore to previous X,Y,M to make the objective nonincreasing
        Xm = X0; Ym = Y0; Yt = Y0'; Ys = Y0*Yt; 
        MYt = M0*Yt; eta = min(1.5*eta,0.5);
    else
        % apply extrapolation
        w = (t0-1)/t; % extrapolation weight
        wx = min([w,rw*sqrt(Lx0/Lx)]); % choose smaller one for convergence
        wy = min([w,rw*sqrt(Ly0/Ly)]);
        Xm = X+wx*(X-X0);    Ym = Y+wy*(Y-Y0); % extrapolation
        X0 = X; Y0 = Y; M0 = M; t0 = t; obj0 = obj;
    end  
end
fprintf('\n');
Out.iter = k; % report # of iterations

%% <http://www.caam.rice.edu/~optimization/bcu/nmfc/nmfc.m Download this m-file>
