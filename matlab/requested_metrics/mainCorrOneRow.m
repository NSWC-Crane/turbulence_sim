% Do Correlation metrics on one row for every 12th row
% Collect metric

%Fix tomorrow - get average of rows for each image !!!!!!


clear
clc

%rangeV = 600:100:1000;
rangeV = [800];
%zoom = [2000, 3000,  4000, 5000];
zoom = [4000];

dirOut = "C:\Data\JSSAP\modifiedBaselines\CorrPlots_OneRow";

for rng = rangeV
    r_matr = [];
    for zm = zoom
        [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfoMod(rng, zm);

        %Read In baseline and i00 real image
        ImgB = imread(fullfile(dirBase, basefileN));
        ImgR = imread(fullfile(dirSharp, ImgNames{1}));
        ImgR = ImgR(:,:,2);  % only green channel
        [m,n] = size(ImgB);
        ctr = uint8(m/2);
        
        row = 1;
        cc = [];
        figure()
        while row < n
            
            lKernel = [1, -2, 1];
            lapImgB = conv(ImgB(row,1:ctr), lKernel, 'same');
            lapImgR = conv(ImgR(row,1:ctr), lKernel, 'same');
            lb_fft = fftshift(fft(lapImgB)/numel(lapImgB));
            lr_fft = fftshift(fft(lapImgR)/numel(lapImgR));
            diff_fft_rb = lr_fft - lb_fft;
            % Autocorrelation/convolution?
            cv_diff_rb = conv(diff_fft_rb, diff_fft_rb(end:-1:1), 'same');
            s_01 = sum(abs(cv_diff_rb(:)));

            % Add in 20 images at same zoom/range (including itself)
            rt = [];
            for i = 1:length(ImgNames)
                ImgOtR = imread(fullfile(dirSharp, ImgNames{i}));
                ImgOtR = ImgOtR(:,:,2);  % only green channel
                lapImgOtR = conv(ImgOtR(row,1:ctr), lKernel, 'same');
                lotr_fft = fftshift(fft(lapImgOtR)/numel(lapImgOtR));
                diff_fft_otRb = lotr_fft - lb_fft;
                % Cross convolution?
                cv_diff_otRb = conv(diff_fft_rb, diff_fft_otRb(end:-1:1), 'same');

                s_02 = sum(abs(cv_diff_otRb(:)));
                r012 = s_02/s_01;
                r = 1-abs(1-r012);

                % Collect ratios by row
                cc = [cc; [row, i, r]];
                rt = [rt r];

            end
            x = 1:20;
            plot(x, rt)
            hold on

            row = row + 20;
        end
        grid on
        legend("1", "21", "41","61", "81", "101","121","141","161","181","201","221","241", 'location','southeast')
        xlim([1,20])
        title("Range " + num2str(rng) + " Zoom " + num2str(zm))
        fileN = fullfile(dirOut,"r" + num2str(rng) + "_z" + num2str(zm) + ".png");
        f = gcf;
        exportgraphics(f,fileN,'Resolution',300)

        %close all;
  
    end
end

figure
hold on
plot(ImgB(row,1:ctr), 'b');
plot(ImgR(row,1:ctr), 'r');
plot(ImgOtR(row,1:ctr), 'g');
legend('ImgB','ImgR','ImgOtR')
%title('data & data-warp1 & data-warp2')

figure
hold on
plot(lapImgB, 'b');
plot(lapImgR, 'r');
plot(lapImgOtR, 'g');
legend('lapImgB','lapImgR','lapImgOtR')

figure
hold on
plot(abs(lb_fft), 'b');
plot(abs(lr_fft), 'r');
plot(abs(lotr_fft), 'g');
legend('lb_fft','lr_fft','lotr_fft')

figure
hold on
plot(abs(diff_fft_rb), 'b');
plot(abs(diff_fft_otRb), 'r');
legend('diff_fft_rb','diff_fft_rb')

figure
hold on
plot(abs(cv_diff_rb), 'b');
plot(abs(cv_diff_otRb), 'r');
legend('cv_diff_rb','cv_diff_otRb')