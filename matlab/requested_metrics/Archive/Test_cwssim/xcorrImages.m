function [sum_ac,sum_cc] = xcorrImages(RefImg, TestImg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Mean was subtracted from image prior to entering function
% Border pixels were removed prior to entering function

% Cross correlate images, not FFT of images
ac = xcorr2(RefImg);
sum_ac = sum(ac, 'all');
cc = xcorr2(RefImg, TestImg);
sum_cc = sum(cc, 'all');




end