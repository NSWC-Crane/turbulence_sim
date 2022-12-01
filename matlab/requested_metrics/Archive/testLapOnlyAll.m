clear
clc

% Set up table with columns zoom, range, cn2, r0 using data file:
% C:\Data\JSSAP\combined_sharpest_images_withCn2FriedPar.xlsx
T1 = readtable('C:\Data\JSSAP\combined_sharpest_images_withAtmos.xlsx','Sheet','combined_sharpest_images', ...
    'Range', 'D1:E55');
T2 = readtable('C:\Data\JSSAP\combined_sharpest_images_withAtmos.xlsx','Sheet','combined_sharpest_images', ...
    'Range','T1:U55');
T = [T1 T2];

% Call function FilterMetrics for each filter and then each range, zoom
% combination
% Setup matrix for average metric
AvgDFL = zeros(54,1);
AvgDFL_all = [];
tT4 = [];
tT1 = [];
tT3 = [];

% Note MetricsF(:,1) is range
% Note MetricsF(:,2) is zoom
for indx = 1: height(T)
        
    
        [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfoMod(T{indx,1}, T{indx,2});

        metricsF = zeros(length(ImgNames),1);

        % Read in baseline image - used modified Baseline
        pathB = fullfile(dirBase, basefileN);
        ImageB = imread(pathB);
        %ImageB = ImageB(:,:,2);  % Use green channel

        % Read in images
        for i = 1:length(ImgNames)
            pathF = fullfile(dirSharp, ImgNames{i});
            Image = imread(pathF);
            Image = Image(:,:,2);  % Use green channel
            [M,N] = size(Image);

    
            k = fspecial('laplacian',0);
            ImageLap = imfilter(double(Image),k,'replicate','conv'); %,'symmetric');
            ImageLapB = imfilter(double(ImageB),k,'replicate','conv'); %,'symmetric');
            fftI=fft2(double(ImageLap),M,N);
            fftB=fft2(double(ImageLapB),M,N);
            %resultF = abs(fftI-fftB);
            resultF = abs(fftI);
            metricsF(i) = sum(sum(resultF))/(M*N);
            
        end
        AvgDFL_all = [AvgDFL_all; metricsF];
        
        AvgDFL(indx) = mean(metricsF);
end
for i = 1:length(ImgNames)
    tT4 = [tT4; T{:,4} ];
    tT3 = [tT3; T{:,3} ];
    tT1 = [tT1; T{:,1} ];
end

symbs = ['*','*','*','*','*','*','*','s','s'];
sz = 8;

figure()
gscatter(log10(T{:,4}), log10(AvgDFL), T{:,1},[],symbs,sz) %, 'filled') %'LineWidth',2)
xlabel('Fried Parameter log_1_0(r_0)')
%ylim([3.55,4.10])
ylabel('log_1_0(Metric)')
%title('Difference of FFT of Laplacian Images')
ylim([3.10,3.45])
title('FFT of Real Image')
% outf = "C:\Projects\JSSAP\MetricsPlots\avgFried\avgDFLap_Modified.png";
% f = gcf;
%exportgraphics(f,outf,'Resolution',300)

figure()
gscatter(log10(tT4), log10(AvgDFL_all), tT1,[],symbs,sz) %, 'filled') %'LineWidth',2)
xlabel('Fried Parameter log_1_0(r_0)')
ylabel('log_1_0(Metric)')
%ylim([3.55,4.10])
%title('Difference of FFT of Laplacian Images')
ylim([3.10,3.45])
title('FFT of Real Image')
