% Get average metrics for all zoom and range values
% Plot average metric to r0
% For each metric, for each zoom and range value, call FilterMetrics
%    function to return metric for 20 values.  Create a matrix with zoom, 
%    range, r0, and average metric value for each metric
% Plot average metric value as a function of r0

% For each zoom and range, pull cn2 and r0 into new Matrix

% Metrics:  'FFT of Diff Laplacian', 'Diff of FFT Laplacian','FFT of Diff LoGs',
% 'Diff of FFT LoGs','Absolute Difference', 'Absolute FFT'
%

clear
clc

% Set up table with columns zoom, range, cn2, r0 using data file:
% C:\Data\JSSAP\combined_sharpest_images_withCn2FriedPar.xlsx
T1 = readtable('C:\Data\JSSAP\combined_sharpest_images_withAtmos.xlsx','Sheet','combined_sharpest_images', ...
    'Range', 'D1:E55');
T2 = readtable('C:\Data\JSSAP\combined_sharpest_images_withAtmos.xlsx','Sheet','combined_sharpest_images', ...
    'Range','T1:U55');
T = [T1 T2];

% To find range/value in table:
% ind = (T.zoom == 2000) & (T.range == 600);
% T{ind,'cn2'}
% T{ind,'r0'}


% Call function FilterMetrics for each filter and then each range, zoom
% combination
% Setup matrices for each metric
AvgFDL = zeros(54,1);
AvgDFL = zeros(54,1);
AvgFDLoG = zeros(54,4);
AvgDFLoG = zeros(54,4);
AvgAbDiff = zeros(54,1);
AvgAbDiffFFT = zeros(54,1);

% Note MetricsF(:,1) is range
% Note MetricsF(:,2) is zoom
for indx = 1: height(T)
    
         [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfo(T{indx,1}, T{indx,2});
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

%         sigma = 0.2:0.2:0.6;
%         [metrics4] = FilterMetrics('Diff of FFT LoGs', dirBase, dirSharp, basefileN, ImgNames, sigma);
%         avgs = mean(metrics4);
%         AvgDFLoG(indx,1:3) = avgs;

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
% gscatter(T{:,4}, AvgAbDiff, MetricsF(:,1),[],symbs, sz) %, 'filled') %'LineWidth',2)
% xlabel('Fried Parameter r_0')
% ylabel('Metric')
% title('Absolute Difference of Images')
% outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\absdiff.png";
% f = gcf;
% exportgraphics(f,outf,'Resolution',300)
% 
figure()
gscatter(log10(T{:,4}), log10(AvgAbDiffFFT), T{:,1},[],symbs, sz) %, 'filled') %'LineWidth',2)
xlabel('Fried Parameter r_0')
ylabel('Metric')
title('Absolute Difference of FFT of Images')
%outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\absdiff.png";
%f = gcf;
%exportgraphics(f,outf,'Resolution',300)

figure()
gscatter((T{:,4}), log10(AvgDFL), T{:,1},[],symbs,sz) %, 'filled') %'LineWidth',2)
xlabel('Fried Parameter r_0')
ylabel('Metric')
title('Difference of FFT of Laplacian Images')
%outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\avgDFLap_Modified.png";
%f = gcf;
%exportgraphics(f,outf,'Resolution',300)

% 
% figure()
% gscatter(T{:,4}, AvgFDL, T{:,1},[],symbs,sz) %, 'filled') %'LineWidth',2)
% xlabel('Fried Parameter r_0')
% ylabel('Metric')
% title('FFT of Difference of Laplacian Images')
% outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\avgFDLap.png";
% f = gcf;
% exportgraphics(f,outf,'Resolution',300)

% for k = 1:2
%     figure()
%     gscatter(T{:,4}, AvgFDLoG(:,k), T{:,1},[],symbs,sz) %, 'filled') %'LineWidth',2)
%     xlabel('Fried Parameter r_0')
%     ylabel('Metric')
%     title("FFT of Difference of LoG Images, Sigma " + num2str(k))
%     outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\avgFDLoG_S" + num2str(k) + ".png";
%     f = gcf;
%     exportgraphics(f,outf,'Resolution',300)
% end

% sigma = 0.2:0.2:0.6;
% for k = 1:3
%     figure()
%     gscatter((T{:,3}), log10(AvgDFLoG(:,k)), T{:,1},[],symbs,sz) %, 'filled') %'LineWidth',2)
%     %xlabel('Fried Parameter r_0')
%     xlabel('Cn_2')
%     ylabel('Metric')
%     title("Difference of FFT of LoG Images, Sigma " + num2str(sigma(k)))
%     outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\avgDFLoG_S_ModifiedSig" + num2str(k) + ".png";
%     f = gcf;
%     %exportgraphics(f,outf,'Resolution',300)
% end
% 
% figure()
% gscatter(T{:,4}, T{:,3}, T{:,1},[],symbs,sz) %, 'filled') %'LineWidth',2)
% xlabel('Cn^2')
% ylabel('Fried Parameter')
% title('Fried Parameter vs Cn^2')
% outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\friedVsCn2.png";
% f = gcf;
% %exportgraphics(f,outf,'Resolution',300)

