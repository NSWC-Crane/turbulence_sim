% Histogram Matchiing of 2 images - baseline image to real image
% Match only green channel

% Steps 
% 1.  Read in baseline image for range, zoom (green channel only)
% 2.  Read in real image for range, zoom (green channel only)
% 3.  Match pixel value histograms using imhistmatch over several bins
% 4.  Display real image and new baseline image
% 5.  Create simulated image for range, zoom using range/cn2 and newly
% created baseline image.
% 6.  Main test is "varying Cn2" values.

clearvars
clc

% Setup directories
platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

% % Range 650, Zoom 2500
%realI = data_root + "sharpest\z2500\0650\" + "image_z02497_f46620_e11187_i00.png";
%baselineI = data_root + "sharpest\z2500\baseline_z2500_r0650.png";

% Range 600, Zoom 3500
realI = data_root + "sharpest\z3500\0600\" + "image_z03497_f47415_e15004_i00.png";
baselineI = data_root + "sharpest\z3500\baseline_z3500_r0600.png";

%refImg = double(imread(realI)); 
refImg = imread(realI); 
refImg= refImg(:,:,2); 
histRef = imhist(refImg);
%imgB = double(imread(baselineI)); 
imgB = imread(baselineI);
imgB= imgB(:,:,2);
histB = imhist(imgB);

% [newB128, hgram128] = imhistmatch(imgB, refImg, 128);
[newB256, hgram256] = imhistmatch(imgB, refImg, 256);

figure
imshow(refImg)
figure
plot(histRef)
figure
imshow(imgB)
figure
plot(histB)

% figure
% imshow(newB128)
% figure
% plot(hgram128)
figure
imshow(newB256)
figure
plot(imhist(newB256))

% CONCLUSTION:  hgram256 looks exactly like histogram of real image.
% Even though new baseline looks incorrect, new simulated images
% will be created with varying Cn2s using this new baseline image.

% Save new baseline image in test folder NewBL
fileN = "C:\Data\JSSAP\testNewBL\r600_z3500\mhist_z3500_r0600.png";
imwrite(newB256,fileN)
fileR = "C:\Data\JSSAP\testNewBL\r600_z3500\ref_z3500_r0600.png";
imwrite(refImg,fileR)
fileB = "C:\Data\JSSAP\testNewBL\r600_z3500\bl_z3500_r0600.png";
imwrite(imgB,fileB)



