function [ssimReal, ssimImg] = SSIM_FFT_SepRealImg(Image1, Image2, dynamicRange)
% Image1 is the reference image
% Image2 is the simulated image

% Evaluate real and imaginary components

% Find FFT2 of both images
fftImg1 = fftshift(fft2(Image1)/numel(Image1));
fftImg2 = fftshift(fft2(Image2)/numel(Image2));
% Get magnitude and phase of each FFT
realfftImg1 = real(fftImg1);
realfftImg2 = real(fftImg2);
imgfftImg1 = imag(fftImg1);
imgfftImg2 = imag(fftImg2);

% imagesc(ifft2(realfftImg1))
% imagesc(ifft2(imgfftImg1))

% SSIM equation
% Check influence of dynamic range!!!
% mean1: mean of Image1
% var1:  variance of Image1
% cov12: covariance of Image1 and Image2
% c1 = (0.01*dynRange)^2
% c2 = (0.03*dynRange)^2
% SSIM = (2*mean1*mean2+c1)*(2*ccov12+c2)/((mean1^2+mean2^2+c1)*(var1+var2+c2))

% Find mean, variance, and covariance of the real components of the FFTs of each image
% [varR1, meanR1] = var(realfftImg1,[],'all');
% [varR2, meanR2] = var(realfftImg2,[],'all');
% ccovR12 = cov(realfftImg1, realfftImg2);
% covR12 = ccovR12(1,2); 
% Find mean, variance, and covariance of the real components of the FFTs of each image
varR1 = var(realfftImg1,[],'all');
meanR1 = sum(realfftImg1, 'all')/numel(realfftImg1);
varR2 = var(realfftImg2,[],'all');
meanR2 = sum(realfftImg2, 'all')/numel(realfftImg2);
ccovR12 = cov(realfftImg1, realfftImg2);
covR12 = ccovR12(1,2); 


% % Find mean, variance, and covariance of the imaginary components of the FFTs of each image
% [varI1, meanI1] = var(imgfftImg1,[],'all');
% [varI2, meanI2] = var(imgfftImg2,[],'all');
% ccovI12 = cov(imgfftImg1, imgfftImg2);
% covI12 = ccovI12(1,2); 
% Find mean, variance, and covariance of the imaginary components of the FFTs of each image
varI1 = var(imgfftImg1,[],'all');
meanI1 = sum(imgfftImg1, 'all')/numel(imgfftImg1);
varI2 = var(imgfftImg2,[],'all');
meanI2 = sum(imgfftImg2, 'all')/numel(imgfftImg2);
ccovI12 = cov(imgfftImg1, imgfftImg2);
covI12 = ccovI12(1,2);

% % Define constants c1 and c2 using dynamic range - 255 is too high
dynRange = dynamicRange;
c1 = (0.01*dynRange)^2;
c2 = (0.03*dynRange)^2;

% Calculate equation - magnitude
% Looked at parts of equation to evaluate influence of c1 and c2 (minimize influence)
% n1 = 2*meanR1*meanR2;
% n2 = 2*covR12;
% d1 = meanR1^2 + meanR2^2;
% d2 = varR1 + varR2;
% % Choose c1 and c2 to minimize influence
% c1 = 1e-8;
% c2 = 1e-6;
numeratorR = (2*meanR1*meanR2 + c1) *(2*covR12 + c2);
denomR = (meanR1^2 + meanR2^2 + c1)*(varR1 + varR2 + c2);
ssimReal = numeratorR/denomR;

% Calculate equation - phase
% ni1 = 2*meanI1*meanI2;
% ni2 = 2*covI12;
% di1 = meanI1^2 + meanI2^2;
% di2 = varI1 + varI2;
% Choose c1 and c2 to minimize influence
%ci1 = 1e-39;
%ci2 = 1e-6;
numeratorI = (2*meanI1*meanI2 + c1) *(2*covI12 + c2);
denomI = (meanI1^2 + meanI2^2 + c1)*(varI1 + varI2 + c2);
ssimImg = numeratorI/denomI;

end