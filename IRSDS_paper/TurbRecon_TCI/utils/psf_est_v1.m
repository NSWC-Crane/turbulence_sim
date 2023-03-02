function out=psf_est_v1(DxL,DyL,DxI,DyI,H,rho,lambda,max_itr)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Point spread function estimation function for version 1. 
% Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
% Copyright 2020
% Purdue University, West Lafayette, In, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DL are the x and y direction gradient of the estimated latent image
% DI are the x and y direction gradient of the observed blurred image
% H is the psf dictionary

%initialize parameters
[~,~,num_of_filters]=size(H);
sizeIm=size(DxL);

%construct y from I
y=reshape(sqrt(abs(DxI).^2+abs(DyI).^2),[sizeIm(1)*sizeIm(2),1]);

%construct A from L and H
A=[];
for i=1:num_of_filters
    A=[A reshape(sqrt(abs(imfilter(DxL,H(:,:,i),'replicate')).^2+abs(imfilter(DyL,H(:,:,i),'replicate')).^2),[sizeIm(1)*sizeIm(2),1])];
end

A=double(A);
y=double(y);

%initialize variables
x=ones(num_of_filters,1);
x=x/sum(x(:));
v=x;
u=zeros(num_of_filters,1);
ele1=A'*A+rho*eye(num_of_filters);
ele2=A'*y;
r=[];

for i=1:max_itr
    
    %update x
    xtilde = v-u;
    x=ele1\(ele2+rho*xtilde);
    
    %update v
    vtilde = x+u;
    
    f = @(a) sum((vtilde-a/rho).*(abs(vtilde-a/rho)>sqrt(2*lambda/rho))) - 1;
    a=fzero(f,0);
    v=max((vtilde-a/rho).*(abs(vtilde-a/rho)>sqrt(2*lambda/rho)),0);

    u      = u + 0.1*(x-v);
    
    r=[r sum((A*v - y).^2)];
end

out = v;

end
