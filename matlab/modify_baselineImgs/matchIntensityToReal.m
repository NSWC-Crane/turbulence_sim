% Method to create baseline images for simulated images by matching
% intensity values of the real images.
% For each set of range/zoom values of sharpest, real images, make a mean
% image.  Then find the max count value between 0 to 100 and between 180 to
% 255.  These 2 values create the new baseline image by changing 1s and
% 255s in the black and white images (green channel only).

% Start with Range 600/Zoom 2000,2500,3000
% Test with "varying Cn2s" and see if the max metric picks the correct image

% 1. Sum all images at same Range/Zoom
% 2. Find counts of pixels at each pixel value
% 3. For the values between 0 to 100, find the pixel value with the highest
% count (for darkest squares).
% 4. For the values between 180 to 255, find the pixel value with the
% highest count (for lightest square).
% 5. Change baseline image (green channel only) to the above 2 values.

% Functions
% 1. Find highest pixel count in range
% 2. Get baseline and real image file locations
% 3. Change baseline image (green channel) and plot
% 4. Sum images of same range/zoom

clear
clc

%rangeV = 600:50:1000;
rangeV = [600];
%zoomV = [2000 2500 3000 3500 4000 5000];
zoomV = [3500 4000 5000];
col1 = repelem(rangeV.', length(zoomV));
col2 = repmat(zoomV.', length(rangeV),1);
RngZm = [col1.' col2];

for indx = 1:length(RngZm)
    [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfo(RngZm(indx,1), RngZm(indx,2));



    % Create vector to find average histogram for real images with same
    % zoom/range
    meanHist = zeros(256, 1);
    for j = 1:length(ImgNames)      
         pathF = fullfile(dirSharp, ImgNames{j});
         Image = imread(pathF);
         Image = Image(:,:,2);  % Use green channel
         %[M,N] = size(Image);
         [counts, binsLoc] = imhist(Image);
         meanHist = meanHist + counts;
    end

    % Find max value in meanHist from 1:100 and from 180:256
    endDrk = 100;
    strtLgt = 180;
    [countDrk,drkPixVal] = max(meanHist(1:endDrk,1));
    drkPixVal = drkPixVal - 1;

    [countLgt,lgtPixVal] = max(meanHist(strtLgt:256,1));
    lgtPixVal = lgtPixVal + strtLgt - 1;

    % Use lgtPixVal and drkPixVal to match baseline image
    % Make histogram
    [m,n] = size(Image);
    newcount = m*n/2;
    newHist = zeros(256,1);
    newHist(drkPixVal) = newcount; % Half pixels are dark
    newHist(lgtPixVal) = newcount; % Half pixels are light

    % Match to baseline image
    % Read in baseline image
    pathB = fullfile(dirBase, basefileN);
    ImageB = imread(pathB);
    ImageB = ImageB(:,:,2);  % Use green channel
    ModImageB = changem(ImageB,drkPixVal,0);
    ModImageB = changem(ModImageB,lgtPixVal,255);

    % Write new modified baseline images 
    % Will use Python to create simulated images
    fileM = "Mod23_" + basefileN;
    dirM = "C:\Data\JSSAP\baselines2023";
    imwrite(ModImageB, fullfile(dirM, fileM));


end