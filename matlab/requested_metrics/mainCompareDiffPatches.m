% Do Correlation metrics on patches from reals that have similar image
% height/widths and similar pixel steps but different r0s.
% 

clearvars
close all
clc

% For comparison use the real image at Range 650 Zoom 5000 (r0 0.0290) as
% the first image.
% Compare image Range 650 Zomm 3500 (r0 0.1277) to see if similar
% 200x200!!! - Can't use
% Compare image Range 600 Zoom 4000 (r0 0.1079) to see if similar

% % tst 150 pixels
% rng1 = 900;
% zm1 = 4000;
% rng2 = 900;  
% zm2 = 4000;

% Set 1
rng1 = 650;
zm1 = 5000;
rng2 = 600;  
zm2 = 4000;

% % Set 2
% rng1 = 900;
% zm1 = 5000;
% rng2 = 700;  
% zm2 = 2500;

platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

% Import first image
[dirModBase, dirReal1, basefileN, ImgNames1] = GetImageInfoMod(data_root, rng1, zm1);
ImgB = double(imread(fullfile(dirModBase, basefileN)));
ImgR = double(imread(fullfile(dirReal1, ImgNames1{1})));
ImgR = ImgR(:,:,2);  % only green channel
% Read in directories/filenames for second image
[~, dirReal2, ~, ImgNames2] = GetImageInfoMod(data_root, rng2, zm2);
% ImgOtR = imread(fullfile(dirReal2, ImgNames2{1}));
% ImgOtR = ImgOtR(:,:,2);  % only green channel

szPatch = 64;
lKernel = 0.25*[0,-1,0;-1,4,-1;0,-1,0];
%lKernel = [1, -2, 1];

dirOut = data_root + "modifiedBaselines\CorrPlots_PatchesOddImg";
 
ccZ = [];
            
[img_h, img_w] = size(ImgB);

% Setup patches - Assume square images so we'll just use the image height (img_h)
numPatches = floor(img_h/szPatch);
remaining_pixels = img_h - (szPatch * numPatches);

if (remaining_pixels == 0)
    remaining_pixels = szPatch;
    numPatches = numPatches - 1;
end

intv = floor(remaining_pixels/(numPatches + 1));

% Collect ratio without Laplacian
cc = [];
% Collect ratio with Laplacian
cc_l = [];
% Identifier for image (1:i00, 2:i01, etc)
index = 1;

% row,col start at intv,intv
for prow = intv:szPatch+intv:img_h-szPatch
    for pcol = intv:szPatch+intv:img_w-szPatch
               
        ImgB_patch = ImgB(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
        ImgR_patch = ImgR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
        
        ImgB_patch = ImgB_patch - mean(ImgB_patch(:));
        ImgR_patch = ImgR_patch - mean(ImgR_patch(:));
        
        lapImgB = conv2(ImgB_patch, lKernel, 'same');
        lapImgR = conv2(ImgR_patch, lKernel, 'same');
        
        b_fft = fftshift(fft2(ImgB_patch)/numel(ImgB_patch));
        r_fft = fftshift(fft2(ImgR_patch)/numel(ImgR_patch));
        diff_fft_rb = r_fft - b_fft;

        lb_fft = fftshift(fft2(lapImgB)/numel(lapImgB));
        lr_fft = fftshift(fft2(lapImgR)/numel(lapImgR));
        diff_fft_lrb = lr_fft - lb_fft;
        
        % Autocorrelation/convolution
        cv_diff_rb = conv2(diff_fft_rb, conj(diff_fft_rb(end:-1:1, end:-1:1)), 'same');
        s_01 = sum(abs(cv_diff_rb(:)));
        
        cv_diff_lrb = conv2(diff_fft_lrb, conj(diff_fft_lrb(end:-1:1, end:-1:1)), 'same');
        sl_01 = sum(abs(cv_diff_lrb(:)));            

        % Add in 20 images at same zoom/range (including itself)
        rt_l = [];
        rt = [];
        for i = 1:length(ImgNames2)
            ImgOtR = imread(fullfile(dirReal2, ImgNames2{i}));
            ImgOtR = ImgOtR(:,:,2);  % only green channel
            

            ImgOtR_patch = ImgOtR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
            ImgOtR_patch = ImgOtR_patch - mean(ImgOtR_patch(:));
            
%                     figure(2)
%                     imagesc(ImgOtR_patch)
            
            lapImgOtR = conv2(ImgOtR_patch, lKernel, 'same');
            
            otr_fft = fftshift(fft2(ImgOtR_patch)/numel(ImgOtR_patch));
            lotr_fft = fftshift(fft2(lapImgOtR)/numel(lapImgOtR));
            
            diff_fft_otRb = otr_fft - b_fft;
            diff_fft_lotRb = lotr_fft - lb_fft;
            
            % Cross correlation
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
%                 x = 0:19;
%                 figure(1)
%                 hold on
%                 plot(x, rt_l)
        
%                 figure(2)
%                 hold on
%                 plot(x, rt)
        
        index = index + 1;

%             row = row + 20;
    end
end

avg_r = mean(cc,1);
avg_rl = mean(cc_l,1);

figure(1)
x = 0:19;
plot(x, avg_rl, '--r')
hold on
plot(x, avg_r, 'b')
grid on
legend(["With Laplacian" , "Without Laplacian"])
xlim([0,19])
title("R" + num2str(rng1) + " Z" + num2str(zm1) + " to R" + num2str(rng2) + " Z" + num2str(zm2) +" Patch Size " + num2str(szPatch))
hold off
fileN = fullfile(dirOut,"r" + num2str(rng1) + "z" + num2str(zm1) + " toR" + num2str(rng2) + "z" + num2str(zm2) + ".png");
f = gcf;
exportgraphics(f,fileN,'Resolution',300)
% hold off
% 
% %         figure(2)
% %         plot(x, avg_r, '--r')
% %         grid on
% %         %legend("1", "21", "41","61", "81", "101","121","141","161","181","201","221","241", 'location','southeast')
% %         xlim([0,19])
% %         title("Range " + num2str(rng) + " Zoom " + num2str(zm))
% %         hold off
%     
%     %close all;
% ccZ = [ccZ; zm avg_rl];


%figure(2)
%plot(x, )


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