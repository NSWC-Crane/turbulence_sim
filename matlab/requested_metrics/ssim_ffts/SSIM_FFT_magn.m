function [ssimMagn] = SSIM_FFT_magn(Image1, Image2)

% Find FFT2 of both images
fftImg1 = fftshift(fft2(Image1)/numel(Image1));
fftImg2 = fftshift(fft2(Image2)/numel(Image2));
% Get magnitude of each FFT
magfftImg1 = abs(fftImg1);
magfftImg2 = abs(fftImg2);

% SSIM equation
% Check influence of dynamic range!!!
% mean1: mean of Image1
% var1:  variance of Image1
% cov12: covariance of Image1 and Image2
% c1 = (0.01*dynRange)^2
% c2 = (0.03*dynRange)^2
% SSIM = (2*mean1*mean2+c1)*(2*ccov12+c2)/((mean1^2+mean2^2+c1)*(var1+var2+c2))

% Find mean, variance, and covariance of the magnitudes of the FFTs of each image
[var1, mean1] = var(magfftImg1,[],'all');
[var2, mean2] = var(magfftImg2,[],'all');
ccov12 = cov(magfftImg1, magfftImg2);
cov12 = ccov12(1,2); 

% % Define constants c1 and c2 using dynamic range - 255 is too high
% dynRange = 255;
% c1 = (0.01*dynRange)^2;
% c2 = (0.03*dynRange)^2;

% Calculate equation
% Looked at parts of equation to evaluate influence of c1 and c2 (minimize influence)
n1 = 2*mean1*mean2;
n2 = 2*cov12;
d1 = mean1^2 + mean2^2;
d2 = var1 + var2;
% Choose c1 and c2 to minimize influence
c1 = 1e-6;
c2 = 1e-6;
numerator = (2*mean1*mean2 + c1) *(2*cov12 + c2);
denom = (mean1^2 + mean2^2 + c1)*(var1 + var2 + c2);
ssimMagn = numerator/denom;

end
