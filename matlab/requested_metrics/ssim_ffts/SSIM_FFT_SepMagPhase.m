function [ssimMag, ssimPhase] = SSIM_FFT_SepMagPhase(Image1, Image2, dynamicRange)
% Image1 is the reference image
% Image2 is the simulated image

% Evaluate magnitude and phase

% Find FFT2 of both images
fftImg1 = fftshift(fft2(Image1)/numel(Image1));
fftImg2 = fftshift(fft2(Image2)/numel(Image2));
% Get magnitude and phase of each FFT
magfftImg1 = abs(fftImg1);
magfftImg2 = abs(fftImg2);
phfftImg1 = angle(fftImg1);
phfftImg2 = angle(fftImg2);

% imagesc(Image1)
% imagesc(ifft2(magfftImg1))

% SSIM equation
% Check influence of dynamic range!!!
% mean1: mean of Image1
% var1:  variance of Image1
% cov12: covariance of Image1 and Image2
% c1 = (0.01*dynRange)^2
% c2 = (0.03*dynRange)^2
% SSIM = (2*mean1*mean2+c1)*(2*ccov12+c2)/((mean1^2+mean2^2+c1)*(var1+var2+c2))

% Find mean, variance, and covariance of the magnitudes of the FFTs of each image
[varM1, meanM1] = var(magfftImg1,[],'all');
[varM2, meanM2] = var(magfftImg2,[],'all');
ccovM12 = cov(magfftImg1, magfftImg2);
covM12 = ccovM12(1,2); 
% Find mean, variance, and covariance of the phases of the FFTs of each image
[varP1, meanP1] = var(phfftImg1,[],'all');
[varP2, meanP2] = var(phfftImg2,[],'all');
ccovP12 = cov(phfftImg1, phfftImg2);
covP12 = ccovP12(1,2); 

% % Define constants c1 and c2 using dynamic range - 255 is too high
dynRange = dynamicRange;
c1 = (0.01*dynRange)^2;
c2 = (0.03*dynRange)^2;

% Calculate equation - magnitude
% Looked at parts of equation to evaluate influence of c1 and c2 (minimize influence)
% n1 = 2*meanM1*meanM2;
% n2 = 2*covM12;
% d1 = meanM1^2 + meanM2^2;
% d2 = varM1 + varM2;
% % Choose c1 and c2 to minimize influence
% c1 = 1e-6;
% c2 = 1e-6;
numeratorM = (2*meanM1*meanM2 + c1) *(2*covM12 + c2);
denomM = (meanM1^2 + meanM2^2 + c1)*(varM1 + varM2 + c2);
ssimMag = numeratorM/denomM;

% Calculate equation - phase
% np1 = 2*meanP1*meanP2;
% np2 = 2*covP12;
% dp1 = meanP1^2 + meanP2^2;
% dp2 = varP1 + varP2;
% Choose c1 and c2 to minimize influence
cp1 = 1e-10;
cp2 = 1e-4;
numeratorP = (2*meanP1*meanP2 + c1) *(2*covP12 + c2);
denomP = (meanP1^2 + meanP2^2 + c1)*(varP1 + varP2 + c2);
ssimPhase = numeratorP/denomP;

end