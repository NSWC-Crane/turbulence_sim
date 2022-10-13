% Find correlation metric for all zoom and range values, comparing real to
% baseline and then all reals at same zoom/range value to baseline
% Plot all zoom values for each range
% THIS SCRIPT DOES NOT USE LAPLACIAN FILTER

%% Start with one zoom/range

format long g
format compact
clc
close all
clearvars

dirPlots = "C:\Data\JSSAP\modifiedBaselines\SimImages3\CorrPlots2";
        
rangeV = 600:50:1000;
%rangeV = [800];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];
%zoom = [4000];


for rng = rangeV
    r_matr = [];
    for zm = zoom
        % Get info for baseline and all real images
        [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfoMod(rng, zm);

        %Read In baseline and i00 real image
        ImgB = imread(fullfile(dirBase, basefileN));
        ImgR = imread(fullfile(dirSharp, ImgNames{1}));
        ImgR = ImgR(:,:,2);
        
        % FFTs
        [M,N] = size(ImgB);
        FFTB = fftshift(fft2(double(ImgB),M,N));
        FFTR = fftshift(fft2(double(ImgR),M,N));
        % Take difference from baseline
        DiffReal = FFTR - FFTB;
        % Autocorrealtion
        CorrReal = xcorr2(DiffReal); 
        % Sum
        s_Real = sum(abs(CorrReal(:)));
        r_vect = [];
        % Get other reals and compare
        for i = 1:length(ImgNames)
            %Read in other real image and get green channel
            ImgOtR = imread(fullfile(dirSharp, ImgNames{i}));
            ImgOtR = ImgOtR(:,:,2);
          
            % FFT of other real
            FFTOtR = fftshift(fft2(double(ImgOtR),M,N));
            % Take difference of baseline FFT from other real FFT
            DiffOthR = FFTOtR - FFTB;
            % Cross Correlation
            CorrOtR = xcorr2(DiffReal, DiffOthR); 
            % Sum
            s_RealOtR = sum(abs(CorrOtR(:)));
            r_RR = s_RealOtR/s_Real;
            r = 1-abs(1-r_RR);
            r_vect = [r_vect; r];

        end

        r_matr = [r_matr r_vect];
    end

    sx = size(r_matr);
    figure();
    
    for i = 1:sx(2)
        plot(1:20, r_matr(:,i))
        hold on
    end
    
    grid on
    xlim([1,20])
    %ylim([0.85,1.00])
    legend('2000','2500','3000','3500','4000','5000','location','southwest')
    title("Reals without Laplacian Filter: " + " Range " + num2str(rng))
    fileN = fullfile(dirPlots,"r" + num2str(rng) + "_RealsWITHOUTLapl.png");
    f = gcf;
    exportgraphics(f,fileN,'Resolution',300)

    close all
end
