
% Histogram matching for each baseline image by zoom and range value using 
% the real, sharpest images as the reference.
% Purpose: Create a baseline image for simulation software that closer
% matches the real, sharpest images.
% Use MATLAB:  look at imhist, imhistmatch, histeq
%               [counts, binLocations] = imhist(Image);
%               [Jimage, htgram] = imhistmatch;
%               [MImage,mhist] = histeq(ImageB, meanHist);
% Goal: Create an intensity-matched baseline image for each zoom/range
% combo.

% 1.  Using 20 sharpest images in each zoom/range set, find average count
% for each bin from 0 to 255 (or 256 bins).
% 2.  Match baseline image histogram for that zoom/range to new avg
% histogram.
% 3. Save new matched baseline image.
% 4.  Run Purdue simulated images on new matched basline image, using only
% blur and then blur/tilt (use Purdue's python code).
% 5.  Compare to real images.

%{
        Baseline image has 2 intensities, 0 and 256.  Matched image using 
        imhistmatch has 2 intensities, 138 and 236.  Don't use 'imhistmatch'.
        Rule:  Matched image:  the 2 pixels values have to have > 5% of 256
        pixels with that pixel value.  5% is 12.8 pixel count.  From low
        intensity side, the first bin (max) that has > 12.8 count is the low
        intensity value.  From the high intensity side, the first bin (min)
        with > 12.8 count is the high intensity value.        
    %}

clear
clc

rangeV = 600:50:1000;
zoomV = [2000 2500 3000 3500 4000 5000];
col1 = repelem(rangeV', length(zoomV));
col2 = repmat(zoomV', length(rangeV),1);
Sets = [col1 col2];

% Find mean histogram for each zoom/range entry.
% Create cell array to hold mean Images for baseline
MImageArray = cell(length(Sets),1);
for indx = 1:length(Sets)
    [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfo(Sets(indx,1), Sets(indx,2));
    % Create vector to find average histogram
    meanHist = zeros(256, 1);
    for j = 1:length(ImgNames)      
         pathF = fullfile(dirSharp, ImgNames{j});
         Image = imread(pathF);
         Image = Image(:,:,2);  % Use green channel
         [M,N] = size(Image);
         [counts, binsLoc] = imhist(Image);
         meanHist = meanHist + counts;
    end
    meanHist = meanHist./length(ImgNames);
    searchHist = meanHist > (0.001 * (M*N));  %% THIS IS ARBITRARY - TEST IT 0.1% of total number of pixels
    indxM =binsLoc(searchHist); %binsLoc is (0,255)
    histM = zeros(256,1);
    histM(indxM(1)+1) = 128;
    histM(indxM(end)+1) = 128;

    % Match to baseline image
    % Read in baseline image
    pathB = fullfile(dirBase, basefileN);
    ImageB = imread(pathB);
    ImageB = ImageB(:,:,2);  % Use green channel
    [MImage,mhist] = histeq(ImageB, histM);

    % Write new modified baseline images 
    % Will use Python to create simulated images
    fileM = "Mod_" + basefileN;
    dirM = "C:\Data\JSSAP\modifiedBaselines";
    imwrite(MImage, fullfile(dirM, fileM));
    MImageArray{indx} = MImage;

%     figure()
%     histogram(meanHist, binsLoc)
%     figure()
%     histogram(counts, binsLoc)
%     figure()
%     histogram(ImageB)
%     figure()
%     histogram(MImage)
%     figure()
%     imshow(MImage)

end

% Look at histogram of new simulated images, new baseline, and real images


% For display, to change edge of histogram:
% histHandle = histogram(data);
% histHandle.BinEdges = histHandle.BinEdges +
% histHandle.BinWidth/2;(make sure not cutoff)