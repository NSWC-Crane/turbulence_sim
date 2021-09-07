function [img_out, ph, outPSF] = sim_fun(img,params,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = sim_fun(y,params,opts)
% 
% This function implements the wave propagation through turbulence
% simulator proposed in Chimitt and Chan "Simulating Anisoplanatic
% Turbulence by Sampling Inter-modal and Spatially Correlated Zernike
% Coefficients". Please visit (enter Optical Engineering edition once 
% it's uploaded there) or arxiv.org/ab/2004.11210 for a pre-print edition.
%
% Additionally, if you find this simulator useful in your work (which is
% our aim in making this open source!) please consider citing
% the paper. Thank you!
%
% Final note. We aim to have a temporal version of this simulator released
% in the future. Currently each frame is considered to be independent 
% from the last. This is acceptable for a wide variety of applications,
% though the limitations are obvious. Extending the results in our work to
% incorporate temporal evolution is required for accurate simulation,
% though if desired, one could implement something as a placeholder for
% more accurate temporal modeling.
%
% This work has been cleared for public release carrying the approval
% number 88ABW-2019-5034
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%                 y    -  input image/sequence (row-col-frames) greyscale
%                           required
%              params  -  object containing all parameters
%     params.t_params  -  object containing turbulence parameters
%     params.t_params.D      -  aperture diameter size (m) {0.2034}
%     params.t_params.lambda -  wavelength (m) {0.525e-6}
%     params.t_params.L      -  propagation length (m) {7000}
%     params.t_params.Cn2    -  index of refraction structure
%                                 parameter {5e-16} (yes, it's that 
%                                 small! 5e-17 to 5e-15 is a good 
%                                 basic range.)
%     params.t_params.d      -  focal length (m) {1.2}
%     params.t_params.k      -  wave number (rad/m) 
%                                 {2*pi/params.t_params.lambda}
%     params.s_params  -  object containing sampling parameters
%     params.s_params.rowsW  -  number of rows in phase (pixels) {64}
%     params.s_params.colsW  -  number of rows in phase (pixels) {64}
%     params.s_params.fftK   -  upsampling ratio {2}
%     params.s_params.K      -  number of PSFs used per row {16}
%     params.s_params.T      -  PSFs in total image {params.s_params.K^2}
%              opts.frames  -  number of frames desired in output {100}                  
%            ** default values of opts are given in {}.
%
%
%Output:          out  -  generated turbulence sequence 
%
%Stanley Chan and Nick Chimitt
%Copyright 2020
%Purdue University, West Lafayette, IN, USA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check inputs
if nargin<1
    error('not enough input, try again \n');
elseif nargin==1
    params = [];
    params.t_params = [];
    params.s_params = [];
    opts = [];
elseif nargin==2
    if ~isfield(params,'t_params')
        params.t_params = [];
    end
    if ~isfield(params,'s_params')
        params.s_params = [];
    end
    opts = [];
elseif nargin==3
    if ~isfield(params,'t_params')
        params.t_params = [];
    end
    if ~isfield(params,'s_params')
        params.s_params = [];
    end
end

% Check defaults for turbulence parameters
if ~isfield(params.t_params,'D')
    params.t_params.D = 0.2034;
end
if ~isfield(params.t_params,'lambda')
    params.t_params.lambda = 0.525e-6;
end
if ~isfield(params.t_params,'L')
    params.t_params.L = 7000;
end
if ~isfield(params.t_params,'Cn2')
    params.t_params.Cn2 = 5e-16;
end
if ~isfield(params.t_params,'d')
    params.t_params.d = 1.2;
end
if ~isfield(params.t_params,'k')
    params.t_params.k = 2*pi/params.t_params.lambda;
end

% Check defaults for sampling parameters
if ~isfield(params.s_params,'rowsW')
    params.s_params.rowsW = 64;
end
if ~isfield(params.s_params,'colsW')
    params.s_params.colsW = 64;
end
if ~isfield(params.s_params,'fftK')
    params.s_params.fftK = 2;
end
if ~isfield(params.s_params,'K')
    params.s_params.K = 16;
end
if ~isfield(params.s_params,'T')
    params.s_params.T = params.s_params.K^2;
end

% Checking for image vs. sequence of images
if length(size(img)) == 2
    in_im = 1;
    if ~isfield(opts,'frames')
        opts.frames = 100;
    end
else
    in_im = 0;
    opts.frames = size(img,3);
end

rng(12345);

% set turbulence parameters
D      = params.t_params.D;
lambda = params.t_params.lambda;
L      = params.t_params.L;
Cn2    = params.t_params.Cn2;
d      = params.t_params.d;
k      = params.t_params.k;

% set sampling parameters
rowsW  = params.s_params.rowsW;
colsW  = params.s_params.colsW;
fftK   = params.s_params.fftK;
K      = params.s_params.K;
T      = params.s_params.T;

% set additional parameters
frames = opts.frames;
rows = size(img,1);
cols = size(img,2);

% Constants in [Chanan 1992]
f0      = @(k) k.^(-14/3).*besselj(2,k).^2;
f1      = @(z) (z/L).^(5/3);
f2      = @(z) (1-z/L).^(5/3);
c1      = 2*((24/5)*gamma(6/5))^(5/6);
c2      = 4*c1/pi*(gamma(11/6))^2;
c3      = integral(f0, 1e-12, 1e3);
c4      = 2.91*k^2*L^(5/3)*Cn2;
r0      = ((0.423*(2*pi/lambda)^2)*Cn2*integral(f1, 0, L))^(-3/5);
theta0  = (c4*integral(f2, 1e-12, L))^(-3/5);
delta0  = L*lambda/(2*D);
deltaf  = d*lambda/(2*D);

W     = circ(rowsW, colsW);
idx_r = rowsW*fftK/2+[-rowsW/2+1:rowsW/2];
idx_c = colsW*fftK/2+[-colsW/2+1:colsW/2];
kappa = sqrt( (D/r0)^(5/3)/(2^(5/3))*(2*lambda/(pi*D))^2 * 2*pi )*L/delta0;
C     = Zernike_GenCoeff;
C     = C(4:36,4:36);
[U,S] = eig(C);
R     = real(U*sqrt(S));


%% generate the bluring point spread functions/kernels
outPSF = zeros(length(idx_r),length(idx_c),T,frames);
for k = 1:frames
    for i=1:T
        if mod(i,10)==0
%             fprintf('i = %3g / %3g \n', i, T);
        end
        b = randn(size(C,1),1);
        a = R*b;
        [ph(:,:,i,k), ~] = ZernikeCalc(4:36, a*kappa, rowsW, 'STANDARD');
        U      = exp(1i*2*pi*ph(:,:,i,k)/2).*W;
        uu     = fftshift(abs(ifft2(U, rowsW*fftK, colsW*fftK)).^2);
        outPSF(:,:,i,k) = uu(idx_r, idx_c)/sum(sum(uu(idx_r, idx_c)));
    end
    fprintf('frames = %3g / %3g \n', k, frames);
end

[rowsH, colsH, T, ~] = size(outPSF);
K       = sqrt(T);
blocklength = rowsH + rows/K;
[x,y] = meshgrid(-blocklength/2+1:blocklength/2, -blocklength/2+1:blocklength/2);
[~,r] = cart2pol(x,y);
weight = exp(-r.^2/(2*(blocklength/4)^2));


mc_first = false;

if(mc_first == false)
    
    %% blur the frame
    num = zeros(rows+rowsH);
    den = zeros(rows+rowsH);
    for k = 1:frames
        if in_im
            img_pad = padarray(img, [rowsH/2, colsH/2], 'symmetric');
        else
            img_pad = padarray(img(:,:,k), [rowsH/2, colsH/2], 'symmetric');
        end
        for i=1:K
    %         fprintf('%3g \n', i);
            for j=1:K
                idx   = (i-1)*K+j;
                idx1  = (i-1)*rows/K+rowsH/2+[-rowsH/2+1:rows/K+rowsH/2];
                idx2  = (j-1)*rows/K+rowsH/2+[-rowsH/2+1:rows/K+rowsH/2];
                block = img_pad(idx1, idx2);
                tmp   = imfilter(block, outPSF(:,:,idx,k), 'symmetric');
                num(idx1, idx2) = num(idx1, idx2) + weight.*tmp;
                den(idx1, idx2) = den(idx1, idx2) + weight;
            end
        end
        out      = num./den;
        img_blur(:,:,k) = out(rowsH/2+1:rows+rowsH/2, colsH/2+1:cols+colsH/2);
        fprintf('blurring frame %3g / %3g \n', k, frames);
    end

    %% apply the motion compensation
    N     = 2*rows;
    smax  = delta0/D*N;
    sset  = linspace(0,delta0/D*N,N);
    f     = @(k) k^(-14/3)*besselj(0,2*sset*k)*besselj(2,k)^2;
    I0    = integral(f, 1e-8, 1e3, 'ArrayValued', true);
    g     = @(k) k^(-14/3)*besselj(2,2*sset*k)*besselj(2,k)^2;
    I2    = integral(g, 1e-8, 1e3, 'ArrayValued', true);   
    [x,y] = meshgrid(1:N,1:N);
    s     = round(sqrt((x-N/2).^2 + (y-N/2).^2));
    s     = min(max(s,1),N);
    C     = (I0(s) + I2(s))/I0(1);
    C(N/2,N/2)= 1;
    C     = C*I0(1)*c2*(D/r0)^(5/3)/(2^(5/3))*(2*lambda/(pi*D))^2*2*pi;

    Cfft   = fft2(C);
    S_half = sqrt(Cfft);
    S_half(S_half<0.002*max(S_half(:))) = 0;
    for i = 1:frames
        MVx    = real(ifft2(S_half.*randn(2*rows,2*cols)))*sqrt(2)*2*rows*(L/delta0);
        MVx    = MVx(rows/2+1:2*rows-rows/2, 1:rows);
        MVy    = real(ifft2(S_half.*randn(2*rows,2*cols)))*sqrt(2)*2*rows*(L/delta0);
        MVy    = MVy(1:cols,cols/2+1:2*cols-cols/2);
        img_out(:,:,i) = MotionCompensate(img_blur(:,:,i),MVx,MVy,0.5);
    end

else
    %% apply the motion compensation
    N     = 2*rows;
    smax  = delta0/D*N;
    sset  = linspace(0,delta0/D*N,N);
    f     = @(k) k^(-14/3)*besselj(0,2*sset*k)*besselj(2,k)^2;
    I0    = integral(f, 1e-8, 1e3, 'ArrayValued', true);
    g     = @(k) k^(-14/3)*besselj(2,2*sset*k)*besselj(2,k)^2;
    I2    = integral(g, 1e-8, 1e3, 'ArrayValued', true);   
    [x,y] = meshgrid(1:N,1:N);
    s     = round(sqrt((x-N/2).^2 + (y-N/2).^2));
    s     = min(max(s,1),N);
    C     = (I0(s) + I2(s))/I0(1);
    C(N/2,N/2)= 1;
    C     = C*I0(1)*c2*(D/r0)^(5/3)/(2^(5/3))*(2*lambda/(pi*D))^2*2*pi;

    Cfft   = fft2(C);
    S_half = sqrt(Cfft);
    S_half(S_half<0.002*max(S_half(:))) = 0;
    for i = 1:frames
        MVx    = real(ifft2(S_half.*randn(2*rows,2*cols)))*sqrt(2)*2*rows*(L/delta0);
        MVx    = MVx(rows/2+1:2*rows-rows/2, 1:rows);
        MVy    = real(ifft2(S_half.*randn(2*rows,2*cols)))*sqrt(2)*2*rows*(L/delta0);
        MVy    = MVy(1:cols,cols/2+1:2*cols-cols/2);
        img_mc(:,:,i) = MotionCompensate(img(:,:),MVx,MVy,0.5);
    end

    %% blur the frame
    num = zeros(rows+rowsH);
    den = zeros(rows+rowsH);
    for k = 1:frames
        if in_im
            img_pad = padarray(img_mc(:,:,k), [rowsH/2, colsH/2], 'symmetric');
        else
            img_pad = padarray(img_mc(:,:,k), [rowsH/2, colsH/2], 'symmetric');
        end
        for i=1:K
    %         fprintf('%3g \n', i);
            for j=1:K
                idx   = (i-1)*K+j;
                idx1  = (i-1)*rows/K+rowsH/2+[-rowsH/2+1:rows/K+rowsH/2];
                idx2  = (j-1)*rows/K+rowsH/2+[-rowsH/2+1:rows/K+rowsH/2];
                block = img_pad(idx1, idx2);
                tmp   = imfilter(block, outPSF(:,:,idx,k), 'symmetric');
                num(idx1, idx2) = num(idx1, idx2) + weight.*tmp;
                den(idx1, idx2) = den(idx1, idx2) + weight;
            end
        end
        out      = num./den;
        img_out(:,:,k) = out(rowsH/2+1:rows+rowsH/2, colsH/2+1:cols+colsH/2);
        fprintf('blurring frame %3g / %3g \n', k, frames);
    end

    % needed for swapping
    %img_out = img_blur;
end

end