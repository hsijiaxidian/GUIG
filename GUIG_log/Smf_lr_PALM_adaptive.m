function [X,output]=Smf_lr_PALM_adaptive(data,support,siz,opts)% with extrapolation with upper bound of eta 1
    m=siz.m; n=siz.n;  k=siz.k; 
    len=length(support);
    
    eta=min(2*len/(m*n),1);
    tol=1e-5; maxit=500; show_progress=1; show_interval=20; 
    lambda=1e2;
    gate_upt=1/2;
    accelerate = 1;

    if isfield(opts,'p_i');                   p_i       = opts.p_i;                  end
    if isfield(opts,'p_i');                   mu       = opts.mu;                  end
    if isfield(opts,'X');                     X         = opts.X;                    end
    if isfield(opts,'lambda');                lambda    = opts.lambda;               end
    if isfield(opts,'gate_upt');         gate_upt       = opts.gate_upt;             end
    if isfield(opts,'eta');                   eta       = opts.eta;                  end
    if isfield(opts,'tol');                   tol       = opts.tol;                  end
    if isfield(opts,'maxit');                 maxit     = opts.maxit;                end
    if isfield(opts,'show_progress');    show_progress  = opts.show_progress;        end
    if isfield(opts,'show_interval');    show_interval  = opts.show_interval;        end    

    
    num_fac=length(p_i);  % Number of factors
    if show_progress
        fprintf('usemex is false \n');
    end
    W = support;% 矩阵补全中的有效区域矩阵
    M = data;

    
    flagTime=tic;
    time=[];  RMSE=[]; obj_all=[];
    fsp = @(X,p) sum((sum(X.^2)).^p); 
%     fval=@(sxp, WXM,lambda) 0.5*lambda*sum(sxp)+ 1/2*sum(sum(WXM.*WXM));
    fval=@(sxp, WXM,lambda) lambda*sum(sxp)+ 1/2*sum(sum(WXM.*WXM));
 
    Xprod = X{1}*X{end};
    WXM = W.*(Xprod-M); 
    sxp=zeros(1,num_fac);
    sxp(1) = fsp(X{1},p_i(1));sxp(2) = fsp(X{end},p_i(2));
    obj_old=fval(sxp, WXM,lambda);%初始化的目标函数值
    obj = obj_old;
    
    X_old=X;   X_m=X;
    lipz_old=ones(1,num_fac);  
    lipz=lipz_old;
    rand_upt=rand(1,maxit);
    x_tol= zeros(1,num_fac);
    t_old=1;
    ind = 1:k;
    backtrackCount = 0;
     
    for i=1:maxit   % 主循环
        if mod(i,2)==1                                     %Update extrapolation weight 
            t = (1+sqrt(1+4*t_old^2))/2;
            ext = (t_old-1)/t;   
            t_old=t; 
        end
        %% lambda ajustment
        if mod(i,1) == 0 % 1 for lp
            lambda = max(0.9*lambda,mu);
        end

        %% main update
        if rand_upt(i)<= gate_upt   % Update X{1}. Note that we shuffle the updating order, which may result in better performance  
            if accelerate == 1
                X_m{1} = X{1}+ext*(X{1}-X_old{1});         % Extrapolation
                X_old{1} = X{1};
            else
                X_m{1} = X{1};
            end
            
            WXM = W.*(X_m{1}*X{end}-M);Grad = WXM*X{end}';
            Grad_desc = X_m{1} - tau0(1)*Grad; ind1 = ind;
%             [X{1},sxp(1),ind]= str_lp(Xtemp,lambda*tau0(1),p_i(1));
            [X{1},sxp(1),ind]= str_log(Grad_desc,lambda*tau0(1));
            DX = X{1} - X_m{1};
            if norm(X{1}-X_old{1},'fro') / norm(X{1},'fro') < tol
                x_tol(1)=1;
            end
        else                       % Update X{end}
            if accelerate == 1
                X_m{end} = X{end}+ext*(X{end}-X_old{end});
                X_old{end} = X{end};
            else
                X_m{end} = X{end};
            end
            
            WXM = W.*(X{1}*X_m{end}-M); Grad = X{1}'*WXM;
            Grad_desc = X_m{end} - tau0(2)*Grad; ind1 = ind;
%             [X{end},sxp(end),ind]=str_lp(Grad_desc',lambda*tau0(2),p_i(end));
            [X{end},sxp(end),ind]= str_log(Grad_desc',lambda*tau0(2));
            X{end} = X{end}';
            DX = X{end} - X_m{end};
            if norm(X{end}-X_old{end},'fro')/norm(X{end},'fro')< tol
                x_tol(end)=1;
            end
        end
       %% Backtracking line search
       WXM=W.*(X{1}*X{end}-M);
       obj=fval(sxp,WXM,lambda);
       while obj>obj_old + real(dot(DX(:),Grad(:))) + norm(DX(:))^2/(2*tau0) && backtrackCount <=10;
           if rand_upt(i)<= gate_upt           %Without Extrapolation
               ind = ind1;X{1}=X_old{1}; 
           else
               ind = ind1;X{end}=X_old{end};
           end
           rand_upt(i+1)=rand_upt(i);
           eta=min(1.2*eta,0.55);
          backtrackCount = backtrackCount + 1;

       else % 如果目标函数值下降
           obj_old=obj; 
           X{1} = X{1}(:,ind);
           X{end} = X{end}(ind,:);
           X_old{1} = X_old{1}(:,ind);
           X_old{end} = X_old{end}(ind,:);
           if mod(i,1)==0                    
              obj_all=[obj_all,obj];
              time =[time, toc(flagTime)];
              if toc(flagTime)>opts.maxtime
                  break
              end
           end
           backtrackCount = 0;
       end 
       %% adaptive step length estimation         
       if (x_tol(1) && x_tol(end))  %prod(x_tol)
           fprintf('The total iteration number is: %i',i);
           break
       end
    end
    
    
    output.obj_all=obj_all;
    output.time=time;
    output.rmse=RMSE;
    
