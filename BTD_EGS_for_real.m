function out_image=BTD_EGS_for_real(Y,opts)

R=opts.R;  
alpha=opts.alpha; % TVx
tau=opts.tau;  % S
lambda=opts.lambda; %low rank X
max_iter=100;
mu=opts.mu;
tol=opts.tol;
beta=opts.beta; % SC光谱连续性

[n1,n2,n3]=size(Y); 
%normD=norm(Y(:),'fro');
Nway=size(Y); N=min(n1,n2);
A=rand(n1,N*R); B=rand(n2,N*R); C=rand(n3,R);

X=zeros(Nway); Q=X; S=zeros(Nway);
R1=zeros(Nway); R2=R1;
Z1=zeros(Nway); Z2=zeros(Nway); Z31=zeros(Nway); Z32=zeros(Nway);
% TV difference
diaga=ones(Nway(3),1); diagb=ones(Nway(3)-1,1);
D3=diag(-diaga)+diag(diagb,1);
D3(end,1)=1;
d3=D3(:,1);
deig3=fft(d3);
eigD3TD3=beta*(abs(deig3).^2);% n3*1
Eny_x_fft   = (abs(psf2otf([+1; -1], [Nway(1),Nway(2),Nway(3)]))).^2  ;
Eny_y_fft   = (abs(psf2otf([+1, -1], [Nway(1),Nway(2),Nway(3)]))).^2  ;
Eny_fft  =  Eny_x_fft + Eny_y_fft;

for iter=1:max_iter
    Xk=X;
    temp_X=1/2*(Y-S+Q+(Z1-Z2)/mu);
     [A,B,C,X]=btd_SC(temp_X,A,B,C,eigD3TD3,R,lambda/(2*mu));
 
    S=prox_l1(Y-X+Z1/mu,tau/mu);

    
    diff_T=diffT2(mu*R1-Z31,mu*R2-Z32,Nway); % tensor or vector
    temp_Q=reshape(mu*X+Z2+diff_T,Nway);
    Q=real(ifftn(fftn(temp_Q)./(mu*Eny_fft+mu)));
    
    [diff_Qx,diff_Qy]=diff2(Q,Nway);
  
    R1=Thres_21(diff_Qx+Z31/mu,alpha/mu);    
    R2=Thres_21(diff_Qy+Z32/mu,alpha/mu);
    
    Z1=Z1+mu*(Y-X-S);
    Z2=Z2+mu*(X-Q);
    Z31=Z31+mu*(diff_Qx-R1);
    Z32=Z32+mu*(diff_Qy-R2);
    mu=min(mu*1.5,1e6);
    err=norm(X(:)-Xk(:),'fro')/norm(Xk(:),'fro');
    %mpsnr=HSIQA(img*255,X*255);
    fprintf('BTD_EGS: iter=%d mu=%f err=%f \n', iter,mu,err);
    if err<tol
        break
    end  
end
 out_image=X;
end

function S=prox_l1(X,lambda)
S=max(X-lambda,0)+min(X+lambda,0);
end


function [ res ] = Thres_21(x, tau)
        v = sqrt(sum(x.^2,3)); % 每通道求和
        v(v==0) = 1;
        % Weighted group sparsity
        res = repmat( max(1 - tau.*(1./(abs(v)+eps)) ./ v, 0), 1, 1, size(x,3) ) .* x;  
        % Group sparsity (without weighted)
      % res = repmat( max(1 - tau ./ v, 0), 1, 1, size(x,3) ) .* x;
end

function diff_T=diffT2(a,b,Nway)

tenX=reshape(a,Nway);
tenY=reshape(b,Nway);

dfx=diff(tenX,1,1);
dfy=diff(tenY,1,2);
dfxT=zeros(Nway);
dfyT=zeros(Nway);
dfxT(1,:,:)=tenX(end,:,:)-tenX(1,:,:);
dfxT(2:end,:,:)=-dfx;
dfyT(:,1,:)=tenY(:,end,:)-tenY(:,1,:);
dfyT(:,2:end,:)=-dfy;
diff_T=dfxT+dfyT;
end

function [Dx,Dy]=diff2(X,Nway)

tenX=reshape(X,Nway);
dfx=diff(tenX,1,1);
dfy=diff(tenX,1,2);
Dx=zeros(Nway); Dy=zeros(Nway);
Dx(1:end-1,:,:)=dfx;
Dx(end,:,:)=tenX(1,:,:)-tenX(end,:,:);
Dy(:,1:end-1,:)=dfy;
Dy(:,end,:)=tenX(:,1,:)-tenX(:,end,:);
end



