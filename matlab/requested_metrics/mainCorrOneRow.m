% Do Correlation metrics on one row for every 12th row
% Collect metric

%Fix tomorrow - get average of rows for each image !!!!!!


clear
clc

%rangeV = 600:100:1000;
rangeV = [600];
%zoom = [2000, 3000,  4000, 5000];
zoom = [5000];

dirOut = "C:\Data\JSSAP\modifiedBaselines\CorrPlots_OneRow";
% dirOut = "C:\Projects\data\turbulence\ModifiedBaselines\CorrPlots_OneRow";

for rng = rangeV
    r_matr = [];
    for zm = zoom
        [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfoMod(rng, zm);

        %Read In baseline and i00 real image
        ImgB = double(imread(fullfile(dirBase, basefileN)));
        ImgR = double(imread(fullfile(dirSharp, ImgNames{1})));
        ImgR = ImgR(:,:,2);  % only green channel
        
        [m,n] = size(ImgB);
        ctr = uint8(n/4);
        
        row = 1;
        index = 1;
        cc = [];
        figure()
        colormap(colorcube(22))
        row_offset2 = 1;
        for row=1:20:m-21
            row_offset = 4;
            for col=1:ctr:n-1
%             for col=1:1
%         while row < n
            
            lKernel = [1, -2, 1];
            
            ImgB_line = ImgB(row+row_offset,col:col+ctr-1);
            ImgR_line = ImgR(row+row_offset,col:col+ctr-1);
            
            ImgB_line = ImgB_line - mean(ImgB_line(:));
            ImgR_line = ImgR_line - mean(ImgR_line(:));
            
            lapImgB = conv(ImgB_line, lKernel, 'same');
            lapImgR = conv(ImgR_line, lKernel, 'same');
            
            b_fft = fftshift(fft(ImgB_line)/numel(ImgB_line));
            r_fft = fftshift(fft(ImgR_line)/numel(ImgR_line));
            diff_fft_rb = r_fft - b_fft;

            lb_fft = fftshift(fft(lapImgB)/numel(lapImgB));
            lr_fft = fftshift(fft(lapImgR)/numel(lapImgR));
            diff_fft_lrb = lr_fft - lb_fft;
            
            % Autocorrelation/convolution?
            cv_diff_rb = conv(diff_fft_rb, conj(diff_fft_rb(end:-1:1)), 'same');
            s_01 = sum(abs(cv_diff_rb(:)));
            
            cv_diff_lrb = conv(diff_fft_lrb, conj(diff_fft_lrb(end:-1:1)), 'same');
            sl_01 = sum(abs(cv_diff_lrb(:)));            

            % Add in 20 images at same zoom/range (including itself)
            rt_l = [];
            rt = [];
            for i = 1:length(ImgNames)
                ImgOtR = double(imread(fullfile(dirSharp, ImgNames{i})));
                ImgOtR = ImgOtR(:,:,2);  % only green channel
                
                ImgOtR_line = ImgOtR(row+row_offset,col:col+ctr-1);
                ImgOtR_line = ImgOtR_line - mean(ImgOtR_line(:));
                
                lapImgOtR = conv(ImgOtR_line, lKernel, 'same');
                
                otr_fft = fftshift(fft(ImgOtR_line)/numel(ImgOtR_line));
                lotr_fft = fftshift(fft(lapImgOtR)/numel(lapImgOtR));
                
                diff_fft_otRb = otr_fft - b_fft;
                diff_fft_lotRb = lotr_fft - lb_fft;
                
                % Cross convolution?
                cv_diff_otRb = conv(diff_fft_rb, conj(diff_fft_otRb(end:-1:1)), 'same');
                s_02 = sum(abs(cv_diff_otRb(:)));                
                
                cv_diff_lotRb = conv(diff_fft_lrb, conj(diff_fft_lotRb(end:-1:1)), 'same');
                sl_02 = sum(abs(cv_diff_lotRb(:)));
                
                r_012 = s_02/s_01;
                r = 1-abs(1-r_012);
                
                rl012 = sl_02/sl_01;
                rl = 1-abs(1-rl012);

                % Collect ratios by row
%                 cc = [cc; [row, i, rl]];
                
                
                rt = [rt r];
                rt_l = [rt_l rl];
                cc(index, i) = rl;

            end
            x = 1:20;
            hold on
            plot(x, rt_l)
%             plot(x, rt, '--b')
            
            index = index + 1;

            row_offset = row_offset + 4;
%             row = row + 20;
            end
        end
        avg_rl = mean(cc,1);
        plot(x, avg_rl, '--r')
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
plot(abs(diff_fft_lrb), 'b');
plot(abs(diff_fft_lotRb), 'r');
legend('diff_fft_rb','diff_fft_rb')

figure
hold on
plot(abs(cv_diff_lrb), 'b');
plot(abs(cv_diff_lotRb), 'r');
legend('cv_diff_rb','cv_diff_otRb')