% Do Correlation metrics on patches of 32x32
% Collect metric

clearvars
clc

rangeV = 600:100:1000;
%rangeV = [1000];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];
%zoom = [5000];

szPatch = 64;
lKernel = 0.25*[0,-1,0;-1,4,-1;0,-1,0];
%lKernel = [1, -2, 1];

dirOut = "C:\Data\JSSAP\modifiedBaselines\CorrPlots_Patches";
ccZ = [];
for rng = rangeV
    figure()
    x= 0:19;
    for zm = zoom
        [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfoMod(rng, zm);

        %Read In baseline and i00 real image
        ImgB = double(imread(fullfile(dirBase, basefileN)));
        % Blur baseline
        %ImgB = imgaussfilt(ImgB,0.5, 'FilterSize',9, 'Padding', 'symmetric');
        ImgR = double(imread(fullfile(dirSharp, ImgNames{1})));
        ImgR = ImgR(:,:,2);  % only green channel
        
        [m,n] = size(ImgB);

        % Setup patches
        numPatches = uint8(m/szPatch) - 1;
        
        intv = uint8((m-(numPatches * szPatch))/(numPatches + 1));
        % row,col start at intv,intv
        cc = [];
        cc_l = [];
        
        index = 1;
        
        for prow = intv:szPatch+intv:numPatches * (szPatch+intv)
            for pcol = intv:szPatch+intv:numPatches * (szPatch+intv)
                       
                ImgB_patch = ImgB(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                ImgR_patch = ImgR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                
                ImgB_patch = ImgB_patch - mean(ImgB_patch(:));
                ImgR_patch = ImgR_patch - mean(ImgR_patch(:));

                %lapImgB = imgaussfilt(ImgB_patch, 0.5, 'FilterSize',9, 'Padding', 'symmetric');
                %lapImgR = imgaussfilt(ImgR_patch, 0.5, 'FilterSize',9, 'Padding', 'symmetric');
                
                lapImgB = conv2(ImgB_patch, lKernel, 'same');
                lapImgR = conv2(ImgR_patch, lKernel, 'same');
                
                b_fft = fftshift(fft(ImgB_patch)/numel(ImgB_patch));
                r_fft = fftshift(fft(ImgR_patch)/numel(ImgR_patch));
                diff_fft_rb = r_fft - b_fft;
    
                lb_fft = fftshift(fft(lapImgB)/numel(lapImgB));
                lr_fft = fftshift(fft(lapImgR)/numel(lapImgR));
                diff_fft_lrb = lr_fft - lb_fft;
                
                % Autocorrelation/convolution?
                cv_diff_rb = conv2(diff_fft_rb, conj(diff_fft_rb(end:-1:1, end:-1:1)), 'same');
                s_01 = sum(abs(cv_diff_rb(:)));
                
                cv_diff_lrb = conv2(diff_fft_lrb, conj(diff_fft_lrb(end:-1:1, end:-1:1)), 'same');
                sl_01 = sum(abs(cv_diff_lrb(:)));            
    
                % Add in 20 images at same zoom/range (including itself)
                rt_l = [];
                rt = [];
                for i = 1:length(ImgNames)
                    ImgOtR = double(imread(fullfile(dirSharp, ImgNames{i})));
                    ImgOtR = ImgOtR(:,:,2);  % only green channel
                    
                    ImgOtR_patch = ImgOtR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    ImgOtR_patch = ImgOtR_patch - mean(ImgOtR_patch(:));
                    
                    %lapImgOtR = imgaussfilt(ImgOtR_patch, 0.5, 'FilterSize',9, 'Padding', 'symmetric');
                    lapImgOtR = conv2(ImgOtR_patch, lKernel, 'same');
                    
                    % Look at withouth Laplacian (otr_fft) and with
                    % Laplacian (lotr_fft)
                    otr_fft = fftshift(fft(ImgOtR_patch)/numel(ImgOtR_patch));
                    lotr_fft = fftshift(fft(lapImgOtR)/numel(lapImgOtR));
                    
                    diff_fft_otRb = otr_fft - b_fft;
                    diff_fft_lotRb = lotr_fft - lb_fft;
                    
                    % Cross convolution?
                    cv_diff_otRb = conv2(diff_fft_rb, conj(diff_fft_otRb(end:-1:1, end:-1:1)), 'same');
                    s_02 = sum(abs(cv_diff_otRb(:)));                
                    
                    cv_diff_lotRb = conv2(diff_fft_lrb, conj(diff_fft_lotRb(end:-1:1, end:-1:1)), 'same');
                    sl_02 = sum(abs(cv_diff_lotRb(:)));
                    
                    r_012 = s_02/s_01;
                    r = 1-abs(1-r_012);
                    cc(index, i) = r;
                    
                    rl012 = sl_02/sl_01;
                    rl = 1-abs(1-rl012);
                    cc_l(index, i) = rl;
    
                    rt = [rt r];
                    rt_l = [rt_l rl];
    
                end
                
                index = index + 1;
    
            end
        end
        
        avg_r = mean(cc,1);
        avg_rl = mean(cc_l,1);

        ccZ = [ccZ; rng zm avg_rl];

        plot(x, avg_rl)
        hold on
    end
    
    grid on
    xlim([0,19])
    legend("2000","2500","3000","3500","4000","5000")
    title("Range " + num2str(rng) + " Patch Size " + num2str(szPatch))
%     fileN = fullfile(dirOut,"r" + num2str(rng)  + ".png");
%     f = gcf;
%     exportgraphics(f,fileN,'Resolution',300)
    hold off;
end

% figure
% for i = 1:
% 
%     x= 0:19;
%     plot(x, avg_rl) %
%     grid on
%     xlim([0,19])
%     title("Range " + num2str(rng) + " Zoom " + num2str(zm) + " Patch Size " + num2str(szPatch))
%     %fileN = fullfile(dirOut,"r" + num2str(rng) + "_z" + num2str(zm) + ".png");
%     %f = gcf;
%     %exportgraphics(f,fileN,'Resolution',300)
%   hold off

% figure
% hold on
% plot(ImgB(row,1:ctr), 'b');
% plot(ImgR(row,1:ctr), 'r');
% plot(ImgOtR(row,1:ctr), 'g');
% legend('ImgB','ImgR','ImgOtR')
% %title('data & data-warp1 & data-warp2')
% 
% figure
% hold on
% plot(lapImgB, 'b');
% plot(lapImgR, 'r');
% plot(lapImgOtR, 'g');
% legend('lapImgB','lapImgR','lapImgOtR')
% 
% figure
% hold on
% plot(abs(lb_fft), 'b');
% plot(abs(lr_fft), 'r');
% plot(abs(lotr_fft), 'g');
% legend('lb_fft','lr_fft','lotr_fft')
% 
% figure
% hold on
% plot(abs(diff_fft_lrb), 'b');
% plot(abs(diff_fft_lotRb), 'r');
% legend('diff_fft_rb','diff_fft_rb')
% 
% figure
% hold on
% plot(abs(cv_diff_lrb), 'b');
% plot(abs(cv_diff_lotRb), 'r');
% legend('cv_diff_rb','cv_diff_otRb')