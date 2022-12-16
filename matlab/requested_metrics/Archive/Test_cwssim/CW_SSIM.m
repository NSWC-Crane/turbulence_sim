function [cwssim] = CW_SSIM(RefImg,TestImg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

K = 1;

ReFFT = fftshift(fft2(RefImg)/numel(RefImg));
TestFFT = fftshift(fft2(TestImg)/numel(TestImg));

magReFFT = abs(ReFFT);
magTestFFT = abs(TestFFT);

sum1 = sum(magReFFT.* magTestFFT, 'all');
num1 = 2*(sum1) + K;
sqRe = sum(magReFFT.* magReFFT,'all');
sqTest = sum(magTestFFT.*magTestFFT,'all');
den1 = sqRe + sqTest + K;
m1 = num1/den1;

sumTop = sum(ReFFT * conj(TestFFT),'all');
num2 = 2* abs(sumTop) + K;
sumBottom = abs(sum(ReFFT * conj(TestImg),'all'));
den2 = 2*sumBottom + K;
m2 = num2/den2;

cwssim = m1 * m2;

end