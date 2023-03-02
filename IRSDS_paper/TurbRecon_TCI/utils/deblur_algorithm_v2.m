function [h,coef] = deblur_algorithm_v2(g, rho, lambda, FB, pcamu, stat, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% deblur_v2 implements the blind deconvolution function described in
% section III.C of our paper. 
% 
% Input:    g      - input image
%           rho    - internal parameter of innerloop ADMM
%           lambda - regularization parameter of innerloop ADMM
%           FB     - dictionary for PSF computed by PCA
%           pcamu  - mean of the dictionary
%           stat   - distribution of the PCA coefficient
%           opts   - used to pass additional parameters
%                  - opts.mode: padding mode
% 
% Output:   h      - estimated point spread function
%           coef   - estimated coefficient
% Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
% Copyright 2020
% Purdue University, West Lafayette, IN, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set Defaults
if ~isfield(opts,'mode')
    opts.mode = 'replicate';
end


% Initialize
mode = opts.mode;
mu   = 2000;
[rowf colf colorf] = size(g);
sizeFB=[sqrt(size(FB,1)), sqrt(size(FB,1)), size(FB,2)];
if colorf>1
    g0 = g;
    g = imadjust(g,[],[],1);
    g = rgb2gray(g);
else
    g0 = g;
    g = imadjust(g,[],[],1);
end
maxrowh = sqrt(size(FB,1));
maxcolh = sqrt(size(FB,1));



% 1. Kernel Estimation
% Build Image pyramid
maxrowh_levels = floor(log2(maxrowh)+sqrt(eps));
maxcolh_levels = floor(log2(maxcolh)+sqrt(eps));
levels         = max(maxrowh_levels, maxcolh_levels);
rowh_levels    = [linspace(1,maxrowh_levels, levels) log2(maxrowh)];
colh_levels    = [linspace(1,maxcolh_levels, levels) log2(maxcolh)];

for itr_out= 2:levels+1
    if mod(itr_out,4)==1
        fprintf('\b\b -');
    end
    if mod(itr_out,4)==2
        fprintf('\b\b \\');
    end
    if mod(itr_out,4)==3
        fprintf('\b\b |');
    end
    if mod(itr_out,4)==0
        fprintf('\b\b /');
    end
    
    % Define current h size
    rowhc = floor(2^rowh_levels(itr_out)+sqrt(eps));
    colhc = floor(2^colh_levels(itr_out)+sqrt(eps));
    
    % Define current f size
    rowfc = floor(rowhc/maxrowh*rowf+sqrt(eps));
    colfc = floor(colhc/maxcolh*colf+sqrt(eps));
    
    
    % Ensure odd size f and h (for alignment)
    if (mod(rowhc,2)==0)&&(itr_out<levels+1)
        rowhc = rowhc+1;
    end
    if (mod(colhc,2)==0)&&(itr_out<levels+1)
        colhc = colhc+1;
    end
    if (mod(rowfc,2)==0)&&(itr_out<levels+1)
        rowfc = rowfc+1;
    end
    if (mod(colfc,2)==0)&&(itr_out<levels+1)
        colfc = colfc+1;
    end

    
    % Resize f for current level
    if itr_out==2
        f = imresize(g, [rowfc, colfc], 'bilinear');
    else
        f = imresize(f, [rowfc, colfc], 'bilinear');
    end
    gc = imresize(g, [rowfc colfc], 'bilinear');
    
    % Main Routine to estimate h and f

    [f, h, coef]   = blind_innerloop(gc, rowhc, colhc, FB, pcamu, rho, lambda, stat, f, itr_out, mode, levels);

end

    h = h/sum(abs(h(:)));
end






function [f, h, coef] = blind_innerloop(g, rowh, colh, FB, pcamu, rho, lambda, stat, f, itr_out, mode, levels)
% Initialization
[D, Dt]      = defDDt;
[rowf, colf] = size(g);
max_itr     = 40;
itr         = 1;
change=inf;
coef = [];

% Select appropriate edges for kernel estimation
[gx gy] = D(g);
gxF  = psf2otf(gx, [rowf, colf]);
gyF  = psf2otf(gy, [rowf, colf]);

gxavg = imfilter(gx, ones(5), 'replicate');
gyavg = imfilter(gy, ones(5), 'replicate');
gavg  = sqrt(abs(gxavg).^2 + abs(gyavg).^2);

gxmvg = imfilter(abs(gx), ones(5), 'replicate');
gymvg = imfilter(abs(gy), ones(5), 'replicate');
gmvg  = abs(gxmvg) + abs(gymvg);

r   = (gavg./(gmvg + 0.5));

while itr<=max_itr && change > 1e-10
    % 0. Preparation
    % 0.1 Calculate mask M
    if itr==1
        t   = 1;
        M   = 0;
        while (t<20)&&(nnz(M)<2*sqrt(rowh*colh))
            tr  = (1-0.9^t)*max(r(:));
            M   = r>tr;
            t = t+1;
        end
    end
    
    % 0.2 shock filter and prepare edge map
    fb = imfilter(f, fspecial('gaussian', [9 9], 1), 'replicate');
    fs = shock(fb, 5, 1);
    [Dxfs,Dyfs] = D(fs);
    T = M.*( sqrt(Dxfs.^2+Dyfs.^2) );
    
    save T.mat T
%     T((T<=0.55 & T>=0.35)) = 0;
       
    if itr==1
        t   = 1;
        T1  = 0;
        while (t<20)&&(nnz(T1)<0.5*sqrt(rowh*colh*rowf*colf))
            ts  = (1-0.9^t)*max(T(:));
            T1  = (T>ts);
            t   = t+1;
        end
    end
    

    
    % 0.3 T2 is used to ignore boundary effects
    T2   = zeros(rowf,colf);
    T2(rowh+1:end-rowh, colh+1:end-colh) = 1;
    
    Dxfs = Dxfs.*(T>ts).*T2;
    Dyfs = Dyfs.*(T>ts).*T2;
    Dxfs_F = psf2otf(Dxfs, [rowf, colf]);
    Dyfs_F = psf2otf(Dyfs, [rowf, colf]);
    
    
    
    % 1. Estimate h
    if itr_out<=levels          % not the last two level
        % 1.1 min_h || Dfs * h - Dg ||^2 + gamma ||h||^2 
        gamma = 1e-2;
        num1 = conj(Dxfs_F).*gxF;
        num2 = conj(Dyfs_F).*gyF;
        den1 =  abs(Dxfs_F).^2;
        den2 =  abs(Dyfs_F).^2;
        h    = real(otf2psf((num1+num2)./(den1+den2+gamma), [rowh, colh]));
        coef = 0;
        h(h<0.1*max(h(:))) = 0; 
        h = h/sum(abs(h(:)));

    elseif itr_out>=levels+1     
        if itr~=1
            h_old=h;
        end
        y = vec(g - otf2psf(psf2otf(f).*psf2otf(reshape(pcamu,[sqrt(size(FB,1)) sqrt(size(FB,1))]),size(f))));
        A = zeros(prod(size(f)),212);
        for i = 1:212
            A(:,i) = vec(otf2psf(psf2otf(f).* psf2otf(reshape(FB(:,i),[sqrt(size(FB,1)) sqrt(size(FB,1))]), size(f))));
        end
        [w] = psf_est_v2(A, y, lambda, rho, stat.mu, stat.sig);
        coef = w;
        h = reshape(FB(:,1:212)*w+pcamu',[sqrt(size(FB,1)) sqrt(size(FB,1))]);
        h(h<0) = 0;
        h = h/sum(h(:));

        if itr~=1
            change = mean((h_old(:)-h(:)).^2);
        end
    end
 
    % 2. Estiamte image
    lambda2 = 0.05;
    % 2.1 Pad images
    switch mode
        case 'circular'
            g_pad = g;
        case 'replicate'
            g_pad = padarray(g, [floor(rowf/2), floor(colf/2)], 'replicate');
        otherwise
            g_pad = padarray(g, [floor(rowf/2), floor(colf/2)], 'symmetric');
    end
    [rowp,colp] = size(g_pad);
    
    % 2.2 Deblurring
    % min_f || h * f - g||^2 + ||Df||^2
    if (any(isnan(h(:))))
        h = zeros(size(h));
        h(ceil(numel(h)/2)) = 1;
    end
    H = psf2otf(h, [rowp, colp]);
    num1 = conj(H).*psf2otf(g_pad, [rowp, colp]);
    den1 = abs(H).^2;
    den2 = abs(psf2otf([1 -1] , [rowp, colp])).^2;
    den3 = abs(psf2otf([1 -1]', [rowp, colp])).^2;   
    num  = num1;
    den  = den1+lambda2*(den2+den3);
    f_tmp = real(otf2psf(num./den, [rowp, colp]));
    
    % 2.3 Remove boundary pads
    switch mode
        case 'circular'
            f = f_tmp;
        case 'replicate'
            f = f_tmp(floor(rowf/2)+1:rowf+floor(rowf/2), floor(colf/2)+1:colf+floor(colf/2), :);
        otherwise
            f = f_tmp(floor(rowf/2)+1:rowf+floor(rowf/2), floor(colf/2)+1:colf+floor(colf/2), :);
    end
    
    % 3. update
    itr = itr + 1;
    tr = tr/1.1;
    ts = ts/1.1;    
end


end



function [D,Dt] = defDDt
D  = @(U) ForwardD(U);
Dt = @(X,Y) Dive(X,Y);
end

function [Dux,Duy] = ForwardD(U)
Dux = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
Duy = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
end

function DtXY = Dive(X,Y)
DtXY = [X(:,end,:) - X(:, 1,:), -diff(X,1,2)];
DtXY = DtXY + [Y(end,:,:) - Y(1, :,:); -diff(Y,1,1)];
end

