clc
clear
rangeV = 750;
zoomV = 4000;

for i =0:19
    xlabelsI{i+1} = "i" +num2str(i);
end

% 'FFT of Diff Laplacian', 'Diff of FFT Laplacian','FFT of Diff LoGs',
% 'Diff of FFT LoGs','Absolute Difference'

% Get file information based on range and zoom values
[dirBase, dirSharp, basefileN, ImgNames] = GetImageInfo(rangeV, zoomV);

%  First Laplacian metric and plot
sigma = 1:1;
[metricsF] = FilterMetrics('FFT of Diff Laplacian', dirBase, dirSharp, basefileN, ImgNames, sigma);
figure()
plot(metricsF,  'LineWidth',2)
xlabel('Image Number')
xticks([1:20])
xlim = ([1,20]);
xticklabels(xlabelsI)
ylabel('Metric')
%legend('Sigma 1', 'Sigma 2', 'Sigma 3', 'Sigma 4')
titleI = "Zoom " + num2str(zoomV) + " Range " + num2str(rangeV) + ", FFT of Laplacian Differences";
title(titleI)

%  Second Laplacian metric and plot
sigma = 1:1;
[metricsF] = FilterMetrics('Diff of FFT Laplacian', dirBase, dirSharp, basefileN, ImgNames, sigma);
figure()
plot(metricsF,  'LineWidth',2)
xlabel('Image Number')
xticks([1:20])
xlim = ([1,20]);
xticklabels(xlabelsI)
ylabel('Metric')
%legend('Sigma 1', 'Sigma 2', 'Sigma 3', 'Sigma 4')
titleI = "Zoom " + num2str(zoomV) + " Range " + num2str(rangeV) + ", Difference of FFT Laplacians";
title(titleI)

%  First LoG metric and plot
sigma = 1:4;
[metricsF] = FilterMetrics('FFT of Diff LoGs', dirBase, dirSharp, basefileN, ImgNames, sigma);
figure()
plot(metricsF,  'LineWidth',2)
xlabel('Image Number')
xticks([1:20])
xlim = ([1,20]);
xticklabels(xlabelsI)
ylabel('Metric')
legend('Sigma 1', 'Sigma 2', 'Sigma 3', 'Sigma 4')
titleI = "Zoom " + num2str(zoomV) + " Range " + num2str(rangeV) + ", FFT of LoG Differences";
title(titleI)

%  Second LoG metric and plot
sigma = 1:4;
[metricsF] = FilterMetrics('Diff of FFT LoGs', dirBase, dirSharp, basefileN, ImgNames, sigma);
plot(metricsF,'DisplayName','metricsF')
figure()
plot(metricsF,  'LineWidth',2)
xlabel('Image Number')
xticks([1:20])
xlim = ([1,20]);
xticklabels(xlabelsI)
ylabel('Metric')
legend('Sigma 1', 'Sigma 2', 'Sigma 3', 'Sigma 4')
titleI = "Zoom " + num2str(zoomV) + " Range " + num2str(rangeV) + ", Difference of FFT LoGs";
title(titleI)

%  Absolute difference metric and plot
sigma = 1:1;
[metricsF] = FilterMetrics('Absolute Difference', dirBase, dirSharp, basefileN, ImgNames, sigma);
figure()
plot(metricsF,  'LineWidth',2)
xlabel('Image Number')
xticks([1:20])
xlim = ([1,20]);
xticklabels(xlabelsI)
ylabel('Metric')
%legend('Sigma 1', 'Sigma 2', 'Sigma 3', 'Sigma 4')
titleI = "Zoom " + num2str(zoomV) + " Range " + num2str(rangeV) + ", Absolute Difference";
title(titleI)
