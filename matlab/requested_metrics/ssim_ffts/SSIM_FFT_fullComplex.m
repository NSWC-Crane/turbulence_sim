function [ssimFC] = SSIM_FFT_fullComplex(Image1, Image2, dynamicRange)
% Image1 is the reference image
% Image2 is the simulated image

% Find FFT2 of both images
fftImg1 = fftshift(fft2(Image1)/numel(Image1));
fftImg2 = fftshift(fft2(Image2)/numel(Image2));

% SSIM equation
% Check influence of dynamic range!!!
% mean1: mean of Image1
% var1:  variance of Image1
% cov12: covariance of Image1 and Image2
% c1 = (0.01*dynRange)^2
% c2 = (0.03*dynRange)^2
% SSIM = (2*mean1*mean2+c1)*(2*ccov12+c2)/((mean1^2+mean2^2+c1)*(var1+var2+c2))

% Find mean, variance, and covariance of the FFTs of each image
%[var1, mean1] = var(fftImg1,[],'all');
var1 = var(fftImg1,[],'all');
mean1 = sum(fftImg1, 'all')/numel(fftImg1);
%[var2, mean2] = var(fftImg2,[],'all');
var2 = var(fftImg2,[],'all');
mean2 = sum(fftImg2, 'all')/numel(fftImg2);
ccov12 = cov(fftImg1, fftImg2);
cov12 = ccov12(1,2); 

% % Define constants c1 and c2 using dynamic range - 255 is too high
dynRange = dynamicRange;
c1 = (0.01*dynRange)^2;
c2 = (0.03*dynRange)^2;

% Calculate equation
% % Looked at parts of equation to evaluate influence of c1 and c2 (minimize influence)
% n1 = 2*mean1*mean2;
% n2 = 2*cov12;
% d1 = mean1^2 + mean2^2;
% d2 = var1 + var2;
% % Choose c1 and c2 to minimize influence
% c1 = 1e-8;
% c2 = 1e-5;
numerator = (2*mean1*mean2 + c1) *(2*cov12 + c2);
denom = (mean1^2 + mean2^2 + c1)*(var1 + var2 + c2);
ssimFC = abs(numerator/denom);

mean_imag_fftImg1 = mean(imag(fftImg1),'all');
max_imag_fftImg1 = max(imag(fftImg1),[],'all');
min_imag_fftImg1 = min(imag(fftImg1),[],'all');
mean_real_fftImg1 = mean(real(fftImg1),'all');
max_real_fftImg1 = max(real(fftImg1),[],'all');
min_real_fftImg1 = min(real(fftImg1),[],'all');

mean_imag_fftImg2 = mean(imag(fftImg2),'all');
max_imag_fftImg2 = max(imag(fftImg2),[],'all');
min_imag_fftImg2 = min(imag(fftImg2),[],'all');
mean_real_fftImg2 = mean(real(fftImg2),'all');
max_real_fftImg2 = max(real(fftImg2),[],'all');
min_real_fftImg2 = min(real(fftImg2),[],'all');



end
