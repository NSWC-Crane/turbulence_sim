function out = wrapper_PnP(I,hnew,rho,lambda,gamma,max_itr,method)
% wrapper of the PnP function for deblurring

sizeh = size(hnew);
off = (sizeh-1)/2+20;
opts.rho     = rho; % 0.05
opts.gamma   = gamma; % 1
opts.max_itr = max_itr; % 20
opts.print   = false;
I_pad=padarray(I,off,'symmetric');
sizef=size(I);
out = PlugPlayADMM_deblur(I_pad,hnew,lambda,method,opts);
out=out(off+1:sizef(1)+off,off+1:sizef(2)+off);
