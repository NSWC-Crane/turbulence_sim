
% This version calculates the metrics using scanning windows.

clearvars
clc


simfile = "C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z2000_c1e12_N1.png";
realfile = "C:\Data\JSSAP\z2000\600\image_z01998_f45902_e14987_i00.png";
rng = 600;
zoom = 2000;

% Collect all information in table
TmL = table;
indT = 1;

[winMetrics, winNormMetrics] = metricsScanWindows(simfile, realfile);

% cstr:  cn2 in filename (used to get r0 later)
cstr = strsplit(simfile,'_c');
cstr = strsplit(cstr{2},'.');
cstr = strsplit(cstr{1},'_');

%% Process output for table

% Calculate mean metric and standard deviation of metric for all patches 
% for this image and save to table TmL
mean_wm = mean(winMetrics,'all'); 
std_wm = std(winMetrics,0,'all');
mean_wmN = mean(winNormMetrics,'all'); 
std_wmN = std(winNormMetrics,0,'all');
TmL(indT,:) = {rng zoom string(cstr) simfile simfile realfile...
    realfile mean_wm std_wm mean_wmN std_wmN};
indT = indT + 1;


%% tables
varnames = {'Range', 'Zoom', 'Cn2str', 'SimDir','SimFilename','RealDir','RealFilename',...
    'MeanWindows', 'STDWindows', 'MeanNormWindows', 'STDNormWindows'};
TmL = renamevars(TmL, TmL.Properties.VariableNames, varnames);
TmL.SimFilename = string(TmL.SimFilename);
TmL.RealFilename = string(TmL.RealFilename);

