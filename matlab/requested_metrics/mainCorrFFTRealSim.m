% Find correlation for all zoom and range values, comparing real to
% baseline and then simulated to baseline
% Plot real and simulated results for each zoom/range on same plot

%% Start with one zoom/range

format long g
format compact
clc
close all
clearvars

dirIn = "C:\Data\JSSAP\modifiedBaselines\SimImages3"; % Contains only green channel images
rangeV = 600:50:1000;
%rangeV = [800];
pairR = [900, 1000,600,650,850,950,800,750,700];
%zoom = [2000, 2500, 3000, 3500, 4000, 5000];
zoom = [2000];
pairZ = [3500];
%pairs = [3500, 4000, 5000, 2000, 2500, 3000];

% DO THIS FOR SOMETHING DEFINITELY DIFFERENT
for j = 1:length(rangeV)
    for i = 1:length(zoom)
        % Read in baseline, real, and simulated(tilt/blur) images
        imgBfile = "Mz"+num2str(zoom(i))+"r"+num2str(rangeV(j))+"_GreenC_Img.png";
        imgRfile = "Mz"+num2str(zoom(i))+"r"+num2str(rangeV(j))+"_GreenC_Real.png";
        imgSfile = "Mz"+num2str(zoom(i))+"r"+num2str(rangeV(j))+"_GreenC_SimImg.png";
        % Read in real image that is not same zoom value - to look at
        % different image.
        imgNotFile = "Mz"'+num2str(pairZ(i))+"r"+num2str(pairR(j))+"_GreenC_Real.png";
        
        ImgB = imread(fullfile(dirIn, imgBfile));
        ImgR = imread(fullfile(dirIn, imgRfile));
        ImgS = imread(fullfile(dirIn, imgSfile));
        ImgN = imread(fullfile(dirIn, imgNotFile));

        % Perform Laplacian on all images
        k = fspecial('laplacian',0);
        ImgLapB = imfilter(double(ImgB),k,'replicate','conv'); 
        ImgLapR = imfilter(double(ImgR),k,'replicate','conv'); 
        ImgLapS = imfilter(double(ImgS),k,'replicate','conv'); 
        ImgLapN = imfilter(double(ImgN),k,'replicate','conv'); 
        
        % FFTs of all images
        [M,N] = size(ImgLapB);
        %[M1,N1] = size(ImgLapN);
        FFTLapB = fftshift(fft2(double(ImgLapB),M,N));
        FFTLapR = fftshift(fft2(double(ImgLapR),M,N));
        FFTLapS = fftshift(fft2(double(ImgLapS),M,N));
        FFTLapN = fftshift(fft2(double(ImgLapN),M,N));

        % Take difference from baseline
        DiffReal = FFTLapR - FFTLapB;
        DiffSim = FFTLapS - FFTLapB;
        DiffNot = FFTLapN - FFTLapB;

        % Get Correlations
        CorrReal = xcorr2(DiffReal);
        CorrSim = xcorr2(DiffSim, DiffReal);
        CorrNot = xcorr2(DiffNot, DiffReal);

        % Sum Ratios to Real
        s_Real = sum(abs(CorrReal(:))); %/(M*N);
        s_Sim = sum(abs(CorrSim(:))); %/(M*N);
        s_Not = sum(abs(CorrNot(:))); %/(M1*N1);


        rRS = s_Sim/s_Real;
        rRealSim(j) = 1-abs(1-rRS);
        rRN = s_Not/s_Real;
        rRealNot(j) = 1-abs(1-rRN);

        % Max values to Real
        mx_Real = max(abs(CorrReal(:)));
        mx_Sim = max(abs(CorrSim(:)));
        mx_Not = max(abs(CorrNot(:)));

        %Ratios
        r_mxSim(j) = mx_Sim/mx_Real;
        r_mxNot(j) = mx_Not/mx_Real;

    end
    

end
dirPlots = "C:\Data\JSSAP\modifiedBaselines\SimImages3\CorrPlots";
fileN = fullfile(dirPlots,"z" + num2str(zoom(i)) + ".png");

figure();
plot(rangeV, rRealSim)
hold on
plot(rangeV, rRealNot)
legend('Simulated', "Different Real z" + pairZ(i),'location','east')
grid on
%ylim([0.70,1.00])
title("Zoom " + num2str(zoom(i)))
f = gcf;
exportgraphics(f,fileN,'Resolution',300)

