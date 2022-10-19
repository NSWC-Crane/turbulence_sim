% Do Correlation metrics on patches
% Collect metrics and plot

clearvars
clc

rangeV = 600:100:1000;
%rangeV = [600];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];
%zoom = [2000, 2500];

% Size of sections of image to measure similarity
szPatch = 64;
% Laplacian kernel
lKernel = 0.25*[0,-1,0;-1,4,-1;0,-1,0];

% Setup data directories
platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

dirOut = data_root + "modifiedBaselines\CorrPlots_Patches";

% Collect all information for No Laplacian case
ccZ = [];
% Collect all information for With Laplacian case
ccZl = [];

% Blur
sigma = 0;

for rng = rangeV   
    for zm = zoom
        [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfoMod(data_root, rng, zm);

        %Read In baseline and i00 real image for this zoom/range
        ImgB = double(imread(fullfile(dirBase, basefileN)));
        % Blur baseline
        if sigma > 0
            ImgB = imgaussfilt(ImgB, sigma, 'FilterSize',9, 'Padding', 'symmetric');
        end
        ImgR = double(imread(fullfile(dirSharp, ImgNames{1})));
        ImgR = ImgR(:,:,2);  % only green channel

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
        % Identifier for each patch in image
        index = 1;
                      
        for prow = intv:szPatch+intv:img_h-szPatch
            for pcol = intv:szPatch+intv:img_w-szPatch
                % Baseline image patch   
                ImgB_patch = ImgB(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                ImgB_patch = ImgB_patch - mean(ImgB_patch(:));
                % Real image i00 patch
                ImgR_patch = ImgR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                ImgR_patch = ImgR_patch - mean(ImgR_patch(:));
                % Get Laplacian of both patches
                lapImgB = conv2(ImgB_patch, lKernel, 'same');
                lapImgR = conv2(ImgR_patch, lKernel, 'same');
                % Get difference in FFTs for non-Laplacian case
                b_fft = fftshift(fft2(ImgB_patch)/numel(ImgB_patch));
                r_fft = fftshift(fft2(ImgR_patch)/numel(ImgR_patch));
                diff_fft_rb = r_fft - b_fft;
                % Get difference in FFTs for Laplacian case
                lb_fft = fftshift(fft2(lapImgB)/numel(lapImgB));
                lr_fft = fftshift(fft2(lapImgR)/numel(lapImgR));
                diff_fft_lrb = lr_fft - lb_fft;

                % Autocorrelation/convolution for non-Laplacian case
                cv_diff_rb = conv2(diff_fft_rb, conj(diff_fft_rb(end:-1:1, end:-1:1)), 'same');
                s_01 = sum(abs(cv_diff_rb(:)));  %Sum
                % Autocorrelation/convolution for Laplacian case
                cv_diff_lrb = conv2(diff_fft_lrb, conj(diff_fft_lrb(end:-1:1, end:-1:1)), 'same');
                sl_01 = sum(abs(cv_diff_lrb(:)));  %Sum      
                    
                % Compare with 20 images at same zoom/range (including itself)
                % rt_l = []; % Collect ratios of this patch for each image for non-Laplacian case
                % rt = []; % Collect ratios of this patch for each image for Laplacian case
                for i = 1:length(ImgNames)
                    % Import another real image of same zoom/range for
                    % similarity test
                    ImgOtR = double(imread(fullfile(dirSharp, ImgNames{i})));
                    ImgOtR = ImgOtR(:,:,2);  % only green channel
                    % Create patch for other real image
                    ImgOtR_patch = ImgOtR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    ImgOtR_patch = ImgOtR_patch - mean(ImgOtR_patch(:));
                    % Create Laplacian image of other real patch
                    lapImgOtR = conv2(ImgOtR_patch, lKernel, 'same');
                    
                    % Look at non-Laplacian (otr_fft) 
                    otr_fft = fftshift(fft2(ImgOtR_patch)/numel(ImgOtR_patch));
                    diff_fft_otRb = otr_fft - b_fft;
                    % Cross correlation/convolution
                    cv_diff_otRb = conv2(diff_fft_rb, conj(diff_fft_otRb(end:-1:1, end:-1:1)), 'same');
                    s_02 = sum(abs(cv_diff_otRb(:))); % Sum
                    r_012 = s_02/s_01;  % Ratio
                    r = 1-abs(1-r_012);
                    cc(index, i) = r;

                    % Look at With Laplacian (lotr_fft) 
                    lotr_fft = fftshift(fft2(lapImgOtR)/numel(lapImgOtR));
                    diff_fft_lotRb = lotr_fft - lb_fft;
                    % Cross correlation/convolution
                    cv_diff_lotRb = conv2(diff_fft_lrb, conj(diff_fft_lotRb(end:-1:1, end:-1:1)), 'same');
                    sl_02 = sum(abs(cv_diff_lotRb(:)));  % Sum
                    rl012 = sl_02/sl_01;  % Ratio
                    rl = 1-abs(1-rl012);
                    cc_l(index, i) = rl;
    
                    %rt = [rt r];
                    %rt_l = [rt_l rl];
    
                end
                
                index = index + 1;
    
            end
        end
        avg_r = mean(cc,1);  % non-Laplacian
        avg_rl = mean(cc_l,1); % Laplacian

        ccZ = [ccZ; rng zm numPatches*numPatches avg_r]; % non-Laplacian
        ccZl = [ccZl; rng zm numPatches*numPatches avg_rl]; % Laplacian
        
    end

end                   

% Create plots
x= 0:19;

% Non-Laplacian
for rn = rangeV
        figure()   
        % Setup legend for figure
        zmleg = [];
        for zm = zoom
            plotview = (ccZ(:,1) == rn() & (ccZ(:,2) == zm));
            plot(x, ccZ(plotview,4:end)) % Laplacian
            hold on
            zmleg = [zmleg; num2str(zm)];    
        end
        grid on
        xlim([0,19])
        legend(zmleg)
        if sigma > 0
            title("Non-Laplacian: Range " + num2str(rn) + " Patch Size " + num2str(szPatch) + " Blur s" +num2str(sigma))
            fileN = fullfile(dirOut,"B1NLr" + num2str(rn)  + ".png");
        else
            title("Non-Laplacian: Range " + num2str(rn) + " Patch Size " + num2str(szPatch))
            fileN = fullfile(dirOut,"NLr" + num2str(rn)  + ".png");
        end
        f = gcf;
        exportgraphics(f,fileN,'Resolution',300)
        hold off;
end

% Laplacian
for rn = rangeV
        figure()   
        % Setup legend for figure
        zmleg = [];
        for zm = zoom
            plotview = (ccZl(:,1) == rn() & (ccZl(:,2) == zm));
            plot(x, ccZl(plotview,4:end)) % Laplacian
            hold on
            zmleg = [zmleg; num2str(zm)];    
        end
        grid on
        xlim([0,19])
        legend(zmleg)
        if sigma > 0
            title("Laplacian: Range " + num2str(rn) + " Patch Size " + num2str(szPatch) + " Blur s" +num2str(sigma))
            fileN = fullfile(dirOut,"B1Lr" + num2str(rn)  + ".png");
        else
            title("Laplacian: Range " + num2str(rn) + " Patch Size " + num2str(szPatch))
            fileN = fullfile(dirOut,"Lr" + num2str(rn)  + ".png");
        end

        f = gcf;
        exportgraphics(f,fileN,'Resolution',300)
        hold off;
end
    

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