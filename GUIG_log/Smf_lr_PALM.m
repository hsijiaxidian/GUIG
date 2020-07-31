function [X,Y,output]=Smf_lr_PALM(data,support,siz,opts)% with extrapolation with upper bound of eta 1
    k=siz.k;
    [m,n] = size(data);

    if isfield(opts,'p_i');                   p_i       = opts.p_i;                  end
    if isfield(opts,'p_i');                   mu       = opts.mu;                    end
    if isfield(opts,'X');                     X         = opts.X;                    end
    if isfield(opts,'X');                     Y         = opts.Y;                    end
    if isfield(opts,'lambda');                lambda    = opts.lambda;               end
    if isfield(opts,'eta');                   eta       = opts.eta;                  end
    if isfield(opts,'tol');                   tol       = opts.tol;                  end
    if isfield(opts,'maxit');                 maxit     = opts.maxit;                end
    if isfield(opts,'show_progress');    show_progress  = opts.show_progress;        end   

    W = support;% 矩阵补全中的有效区域矩阵
    M = data;
    t0 = opts.t0;

    RMSE = zeros(maxit, 1); Time = zeros(maxit, 1);
    fsp = @(X,p) sum((sum(X.^2)).^p); 
    fval=@(sxp, WXM,lambda) 0.5*lambda*sum(sxp)+ 1/2*sum(WXM.^2);
%     fval=@(sxp, WXM,lambda) lambda*sum(sxp)+ 1/2*sum(sum(WXM.*WXM));
    [row,col,value] = find(data);
    Prod = partXY(X',Y,row,col,length(value))';
    data_err = Prod - value;
    sxp=zeros(1,2);
    sxp(1) = fsp(X,p_i(1));sxp(2) = fsp(Y,p_i(2));
    obj_old=fval(sxp, data_err,lambda);%初始化的目标函数值
    obj = obj_old;
    
    X_old=X;   Y_old = Y; X_m=X; Y_m = Y;
    lipz_old=ones(1,2);  
    lipz=lipz_old;
    x_tol= zeros(1,2);
    t_old=1;
    ind = 1:k;
    backtrackCount = 0;
    stol = false;
     
    for i=1:maxit   % 主循环                                     %Update extrapolation weight                               
        lambda = max(0.85*lambda,mu);
        t = (1+sqrt(1+4*t_old^2))/2;
        ext = (t_old-1)/t;
        t_old=t;
        %% update X fixign Y
        ext_st = min(ext,sqrt(lipz_old(1)/lipz(1)));
        X_m = X+ext_st*(X-X_old);         % Extrapolation
        X_old = X;
        lipz_old(1) = lipz(1);
        
        Prod = partXY(X_m',Y,row,col,length(value))';
        data_err = Prod - value;
        WXM = sparse(row,col,data_err,m,n);
        
        lipz_temp = norm(Y*Y','fro');
%                      lipz_temp = max(eig(Y*Y'));
        lipz_temp = max(lipz_temp,1e-4);
        lipz(1) = eta*lipz_temp;                      %Muliply eta <1 to get a locally  Lipchitz constants for better performance
        Xtemp=X_m-WXM*Y'/lipz(1);
        ind1 = ind;
%         [X,sxp(1),ind]= str_lp(Xtemp,lambda/lipz(1),p_i(1));
        [X,sxp(1),ind]= str_log(Xtemp,lambda/lipz(1));
        x_tol(1) = false;
        if norm(X-X_old,'fro') / norm(X,'fro') < tol
            x_tol(1)=true;
        end
        X = X(:,ind);
        Y = Y(ind,:);
        X_old = X_old(:,ind);
        Y_old = Y_old(ind,:);
        
        %% update Y fixing X
        ext_end = min(ext,sqrt(lipz_old(end)/lipz(end)));
        Y_m = Y +ext_end*(Y-Y_old);
        Y_old = Y;
        lipz_old(end) = lipz(end);
        
        Prod = partXY(X',Y_m,row,col,length(value))';
        data_err = Prod - value;
        WXM = sparse(row,col,data_err,m,n);        
        lipz_temp = norm(X'*X,'fro');
%                     lipz_temp = max(eig(X'*X));
        lipz_temp = max(lipz_temp,1e-4);
        lipz(end) = eta*lipz_temp;
        Ytemp = Y_m- X'* WXM/lipz(end);
        ind1 = ind;
%         [Y,sxp(end),ind]=str_lp(Ytemp',lambda/lipz(end),p_i(end));
        [Y,sxp(end),ind]= str_log(Ytemp',lambda/lipz(end));
        Y = Y';
        x_tol(end)=false;
        relerr = norm(Y-Y_old,'fro')/norm(Y,'fro');
        if relerr < tol
            x_tol(end)=true;
        end

       %% Stoping cretirien
        
       if (x_tol(1) && x_tol(end)||stol)  %prod(x_tol)
           fprintf('The total iteration number is: %i\n',i);
           break
       end

       if  i <500 %&&(mod(i,10)==0)
           Prod = partXY(X',Y,row,col,length(value))';
           data_err = Prod - value;
           RMSE(i) = MatCompRMSE(X, Y', opts.I_test, opts.J_test, opts.data_test);
           Time(i) = toc(t0);
           if i > 20
           plot(Time(20:i),RMSE(20:i));grid on; drawnow;
           end
           obj=fval(sxp,data_err,lambda);                 
        end 
%         obj_all=[obj_all,obj];
%         plot(1:i,obj_all);drawnow;
      
       if obj> obj_old && backtrackCount<=10              % 如果目标函数值不下降
           ind = ind1;X = X_old; Y=Y_old;
           eta=min(1.2*eta,0.85);
           backtrackCount = backtrackCount +1;
       else % 如果目标函数值下降
           obj_old=obj; 
           X = X(:,ind);
           Y = Y(ind,:);
           X_old = X_old(:,ind);
           Y_old = Y_old(ind,:);
           backtrackCount = 0;

       end 

    end
    
    
    output.RMSE=RMSE(1:i);output.time = Time(1:i);
    