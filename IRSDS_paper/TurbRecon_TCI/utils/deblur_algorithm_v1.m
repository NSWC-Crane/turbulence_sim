function [h,coef] = deblur_algorithm_v1(g, rho, lambda, FB, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% deblur_v1 implements the blind deconvolution function described in
% section III.C of our paper, with the dictionary replaced by gaussians.  
% 
% Input:    g      - input image
%           rho    - internal parameter of innerloop ADMM
%           lambda - regularization parameter of innerloop ADMM
%           FB     - dictionary for PSF computed by PCA
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
[rowf, colf, colorf] = size(g);
sizeFB=size(FB);
if colorf>1
    g0 = g;
    g = imadjust(g,[],[],1);
    g = rgb2gray(g);
else
    g0 = g;
    g = imadjust(g,[],[],1);
end
maxrowh = sizeFB(1);
maxcolh = sizeFB(2);



% 1. Kernel Estimation and Coarse Image Restoration
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
    if itr_out==levels
        for i=1:sizeFB(3)
            resized_FB(:,:,i) = imresize(FB(:,:,i), [rowhc, colhc], 'bilinear');
            resized_FB(:,:,i) = resized_FB(:,:,i)/sum(sum(resized_FB(:,:,i)));
        end
        [f h coef]   = blind_innerloop(gc, rowhc, colhc, resized_FB, rho, lambda, f, itr_out, mode, levels);
    else
        [f h coef]   = blind_innerloop(gc, rowhc, colhc, FB, rho, lambda, f, itr_out, mode, levels);
    end
    
    h = h/sum(abs(h(:)));

end
end






function [f h coef] = blind_innerloop(g, rowh, colh, FB, rho, lambda, f, itr_out, mode, levels)
% Initialization
[D Dt]      = defDDt;
[rowf colf] = size(g);
max_itr     = 40;
itr         = 1;
sizeFB = size(FB);
change=inf;

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

while itr<=max_itr && change > 1e-4
    if mod(itr,4)==1
        fprintf('\b\b -');
    end
    if mod(itr,4)==2
        fprintf('\b\b \\');
    end
    if mod(itr,4)==3
        fprintf('\b\b |');
    end
    if mod(itr,4)==0
        fprintf('\b\b /');
    end
    
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
    if itr_out<levels          % not the last two level
        % 1.1 min_h || Dfs * h - Dg ||^2 + gamma ||h||^2 
        gamma = 1e-2;
        num1 = conj(Dxfs_F).*gxF;
        num2 = conj(Dyfs_F).*gyF;
        den1 =  abs(Dxfs_F).^2;
        den2 =  abs(Dyfs_F).^2;
        h    = real(otf2psf((num1+num2)./(den1+den2+gamma), [rowh, colh]));
        coef = 0;
        h = h/sum(abs(h(:)));

    elseif itr_out>=levels     
        if itr~=1
            h_old=h;
        end
        coef = psf_est_v1(Dxfs,Dyfs,gx,gy,FB,rho,lambda,100);
        coef = coef/sum(coef);
        
        h=zeros(sizeFB(1:2));
        for i=1:sizeFB(3)
            h=h+coef(i)*FB(:,:,i);
        end
        h = h/sum(abs(h(:)));
        
        lbd = 0.001;
        FB=PSFUpdate(FB,lbd,coef,h,0.001,5);       

        if itr~=1
            change = sum(sum(abs(h_old-h)));
        end
    end
    
    
    
    % 2. Estiamte Image
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

