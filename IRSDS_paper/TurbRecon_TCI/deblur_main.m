function [out, h_est] = deblur_main(I, para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [out h_est] = deblur_main(I, para)
% This function is the main routine for the blind deconvolution part
% (section III.C) in the paper
%
% Input: I    - Blurred image (Lucky region fusion result)
%        para - Deblur parameters (provided in the parameters folder)
%             - para.lambda   regularization parameters for inner loop ADMM
%             - para.rho      internal parameters for inner loop ADMM
%             - para.version  version of deblur algorithms to use. 
%                               version1: use gaussians mixtures as
%                               dictionary
%                               version2: use trained PCA weight and
%                               statistics to construct dictionary
%             - para.lambda2  regularization parameters for final
%                             non-blind deblurring using PnP ADMM. 
%             - para.rho2     internal parameters for final
%                             non-blind deblurring using PnP ADMM. 
%
% Output: out   - Recovered image 
%         h_est - Estimated point spread function
%
% Reference: The final non-blind deconvolution is done using: S. H. Chan, X. Wang,
%            and O. Elgendy, "Plug-and-Play ADMM for image restoration: Fixed 
%            point convergence and applications", IEEE Trans. Comp. Imaging, 2017. 
%
% Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
% Copyright 2020
% Purdue University, West Lafayette, IN, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda = para.lambda;
rho = para.rho;
version = para.version;
rho2 = para.rho2;
lambda2 = para.lambda2;
if isfield(para,'dic_range')
    range = para.dic_range;
else
    range = 2.5;
end

% Some fixed PnP Deblur parameters
gamma = 1;
max_itr = 20;
method = 'BM3D';

switch version
    case 1
        dic_num = 50;
        sigma = 0.2:(range-0.2)/(dic_num-1):range;
        dic_size = [31 31];
        H = zeros([dic_size dic_num]);
        for i=1:dic_num
            H(:,:,i)=fspecial('gaussian',dic_size,sigma(i));
        end
        opts.mode='symmetric';
        [h_est,~] = deblur_algorithm_v1(I,rho,lambda,H,opts);
        out = wrapper_PnP(I,h_est,rho2,lambda2,gamma,max_itr,method);
    case 2
        load('./utils/dictionary.mat')
        load('./utils/stat.mat')
        H = coeff;
        clear coeff;
        opts = [];
        [h_est,~] = deblur_algorithm_v2(I,rho,lambda,H,mu,stat,opts);
        out = wrapper_PnP(I,h_est,rho2,lambda2,gamma,max_itr,method);
end

end



