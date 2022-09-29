% Get average metrics for all zoom and range values
% calculate r0 Fried parameter for all zoom and range values
% Plot average metric to r0
% For each metric, for each zoom and range value, call FilterMetrics
% function to return metric for 20 values.  Create a matrix with zoom, 
% range, r0, and average metric value for each metric
% Plot average metric value as a function of r0

% Add r0 to Excel spreadsheet 
% For each zoom and range, pull cn2 and r0 into new Matrix
% Then add columns for each metric in new matrix

% Metrics:  'FFT of Diff Laplacian', 'Diff of FFT Laplacian','FFT of Diff LoGs',
% 'Diff of FFT LoGs','Absolute Difference'
% C:\Data\JSSAP\combined_sharpest_images_withCn2.xlsx

clear
clc

rangeV = transpose(600:50:1000);
zoomV = [2000;2500;3000;3500;4000;5000];
col1 = repelem(rangeV, length(zoomV));
col2 = repmat(zoomV, length(rangeV),1);
MetricsF = [col1 col2];

% For columns 3 and 4, read in cn2 and r0 columns in file named  
% C:\Data\JSSAP\combined_sharpest_images_withCn2.xlsx
col34 = xlsread('C:\Data\JSSAP\combined_sharpest_images_withCn2FriedPar.xlsx','combined_sharpest_images_withCn','N:O');
MetricsF = [MetricsF col34];

% Call function FilterMetrics for each filter and then each range, zoom
% combination
% Setup matrices for each metric
AvgFDL = zeros(54,1);
AvgDFL = zeros(54,1);
AvgFDLoG = zeros(54,2);
AvgDFLoG = zeros(54,2);
AvgAbDiff = zeros(54,1);
AvgAbDiffFFT = zeros(54,1);

% Note MetricsF(:,1) is range
% Note MetricsF(:,2) is zoom
for indx = 1: length(MetricsF)
    
         [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfo(MetricsF(indx,1), MetricsF(indx,2));
%         sigma = 1:1;
%         [metrics1] = FilterMetrics('FFT of Diff Laplacian', dirBase, dirSharp, basefileN, ImgNames, sigma);
%         AvgFDL(indx) = mean(metrics1);
% 
         sigma = 1:1;
         [metrics2] = FilterMetrics('Diff of FFT Laplacian', dirBase, dirSharp, basefileN, ImgNames, sigma);
         AvgDFL(indx) = mean(metrics2);

%         sigma = 1:4;
%         [metrics3] = FilterMetrics('FFT of Diff LoGs', dirBase, dirSharp, basefileN, ImgNames, sigma);
%         avgs = mean(metrics3);
%         AvgFDLoG(indx,1:4) = avgs;

        sigma = 1:2;
        [metrics4] = FilterMetrics('Diff of FFT LoGs', dirBase, dirSharp, basefileN, ImgNames, sigma);
        avgs = mean(metrics4);
        AvgDFLoG(indx,1:2) = avgs;

%         sigma = 1:1;
%         [metrics5] = FilterMetrics('Absolute Difference', dirBase, dirSharp, basefileN, ImgNames, sigma);
%         AvgAbDiff(indx) = mean(metrics5);

        sigma = 1:1;
        [metrics6] = FilterMetrics('Absolute FFT', dirBase, dirSharp, basefileN, ImgNames, sigma);
        AvgAbDiffFFT(indx) = mean(metrics6);

end

% Plots for Abs Diff, Diff FFT of Lap, FFT of Diff Lap, 
% Plots for sigmas of Diff FFT of LoGs and FFT of Diff LoGs
symbs = ['*','*','*','*','*','*','*','s','s'];
sz = 8;

% figure()
% gscatter(MetricsF(:,4), AvgAbDiff, MetricsF(:,1),[],symbs, sz) %, 'filled') %'LineWidth',2)
% xlabel('Fried Parameter r_0')
% ylabel('Metric')
% title('Absolute Difference of Images')
% outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\absdiff.png";
% f = gcf;
% exportgraphics(f,outf,'Resolution',300)
% 
figure()
gscatter(MetricsF(:,4), AvgAbDiffFFT, MetricsF(:,1),[],symbs, sz) %, 'filled') %'LineWidth',2)
xlabel('Fried Parameter r_0')
ylabel('Metric')
title('Absolute Difference of FFT of Images')
outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\absdiff.png";
f = gcf;
exportgraphics(f,outf,'Resolution',300)

figure()
gscatter(MetricsF(:,4), AvgDFL, MetricsF(:,1),[],symbs,sz) %, 'filled') %'LineWidth',2)
xlabel('Fried Parameter r_0')
ylabel('Metric')
title('Difference of FFT of Laplacian Images')
outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\avgDFLap_Modified.png";
f = gcf;
exportgraphics(f,outf,'Resolution',300)
% 
% figure()
% gscatter(MetricsF(:,4), AvgFDL, MetricsF(:,1),[],symbs,sz) %, 'filled') %'LineWidth',2)
% xlabel('Fried Parameter r_0')
% ylabel('Metric')
% title('FFT of Difference of Laplacian Images')
% outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\avgFDLap.png";
% f = gcf;
% exportgraphics(f,outf,'Resolution',300)

% for k = 1:2
%     figure()
%     gscatter(MetricsF(:,4), AvgFDLoG(:,k), MetricsF(:,1),[],symbs,sz) %, 'filled') %'LineWidth',2)
%     xlabel('Fried Parameter r_0')
%     ylabel('Metric')
%     title("FFT of Difference of LoG Images, Sigma " + num2str(k))
%     outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\avgFDLoG_S" + num2str(k) + ".png";
%     f = gcf;
%     exportgraphics(f,outf,'Resolution',300)
% end

for k = 1:2
    figure()
    gscatter(MetricsF(:,4), AvgDFLoG(:,k), MetricsF(:,1),[],symbs,sz) %, 'filled') %'LineWidth',2)
    xlabel('Fried Parameter r_0')
    ylabel('Metric')
    title("Difference of FFT of LoG Images, Sigma " + num2str(k))
    outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\avgDFLoG_S_Modified" + num2str(k) + ".png";
    f = gcf;
    exportgraphics(f,outf,'Resolution',300)
end

figure()
gscatter(MetricsF(:,3), MetricsF(:,4), MetricsF(:,1),[],symbs,sz) %, 'filled') %'LineWidth',2)
xlabel('Cn^2')
ylabel('Fried Parameter')
title('Fried Parameter vs Cn^2')
outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\friedVsCn2.png";
f = gcf;
exportgraphics(f,outf,'Resolution',300)

