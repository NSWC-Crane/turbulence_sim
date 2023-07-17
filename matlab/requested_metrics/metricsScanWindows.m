function [winMetrics, winNormMetrics] = metricsScanWindows(simfilename, realfilename)

% Read real image and define border and window size       
imgR = double(imread(realfilename));
vImgR= imgR(:,:,2);  % Only green channel
[img_h, img_w] = size(vImgR);
removeBorder = 5;
img_hN = img_h-2*removeBorder;
%img_wN = img_w-2*removeBorder;

% Perform metrics by collecting metric from scanning windows
% Size of window
winSize = 16; %(winSize by winSize)
pixScan = 1;
lastPix = img_hN-winSize + 1; % Allow last scan to be complete
firstPix = removeBorder + 1;

%% Processing 2 real images (1: subtract mean, 2: normalize image)
% Remove border
ImgRP = vImgR(firstPix:img_w-removeBorder,firstPix:img_w-removeBorder);
% Subtract mean
ImgRP = ImgRP - mean(ImgRP,"all");
% Normalize: sub
ImgRN = ImgRP./std(ImgRP,0,"all");

% Read in a simulated image in namelist
ImgSim = double(imread(simfilename)); % Sim Image

%% Preprocess 2 simulated image (1: subtract mean, 2: normalize image)
%Remove border
ImgSim = ImgSim(firstPix:img_w-removeBorder,firstPix:img_w-removeBorder);
% Subtract mean
ImgSim = ImgSim - mean(ImgSim,"all");
% Normalized image
ImgSimN = ImgSim./std(ImgSim,0,"all");
   
% Collect metrics of windows 
if pixScan == 1
    winMetrics = ones(lastPix,lastPix);
end

%% Windows
% Running in paralllel:  parfor loop only can use pixSxan = 1 (1:pixScan:lastPix
% parfor col = 1:lastPix
for col = 1:lastPix
    % Create array of row images for current col
    % Vectorized row windows to eliminate one for loop
    winImgR = CreateWindowArray(ImgRP,winSize,col,pixScan, lastPix);
    winImgSim = CreateWindowArray(ImgSim,winSize,col,pixScan, lastPix);
    % Vectorized row windows of normalized image for one column
    winImgRN = CreateWindowArray(ImgRN,winSize,col,pixScan, lastPix);
    winImgSimN = CreateWindowArray(ImgSimN,winSize,col,pixScan, lastPix);                       

    m = turbulence_metric_Array(winImgR, winImgSim);
    winMetrics(:,col) = m.';
    mn = turbulence_metric_Array(winImgRN, winImgSimN);
    winNormMetrics(:,col) = mn.';

end

end