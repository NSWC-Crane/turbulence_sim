% Establish reference image(s) - real or simulated or maybe just directory
%   Give choice of real or simulated?
% Establish test images - real or simulated
%    Give choices
% Plots based on type of method run - real on real , sim on real, sim on
% sim???

format long g
format compact
clc
clearvars

rangeV = 600:50:1000;
%rangeV = [650];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];
%zoom = [2000,2500];

%% OPTIONS
% CALCULATING METRICS
% When calculating metrics, determine if one or multiple patches will be
% used.
onePatch = true;  % Creates only one large patch if true
% Determine if image mean should be subtracted from image prior to
% Laplacian/FFT calculations
subtractMean = true; 
% Determine if Laplacian kernel should be applied to image prior to FFT
useLaplacian = false;

% REFERENCE/TEST IMAGE INFO
% Determine how many reference files wil be used for each range/zoom value
oneReference = true;  % Compare test images to one reference or to multiple references
% Determine if reference file(s) is a real image or a simulated image
realReferences = true;
% Determine if test images are real images or simulated images
realTestImages = false;

% PLOT INFO
% Determine if plots should be saved
savePlots = true;

%% Directories
platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

% Reference file locations
% Only need to define dirRef if references are simulated
% Use GetRealImageFilenames.m if reference files are real
if realReferences == false
    dirRef = data_root + "modifiedBaselines\NewSimulations\SimReal\";
end

% Test Image Locations


% Output directory for plots



