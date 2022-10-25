% For each range, zoom value perform similarity metrics on files in the
% directory C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2
% Similarity metrics between one of the 20 sharpest images and all
% simulated images created with varying cn2/r0 values

clearvars
clc

rangeV = 600:100:1000;
%rangeV = [600];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];
%zoom = [2000, 2500];

platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

% Define directories
% Location of simulated images by Cn2 value
dirSims = data_root + "modifiedBaselines\SimImgs_VaryingCn2";
% Location to save plots
dirOut = data_root + "modifiedBaselines\SimImgs_VaryingCn2\Plots";

% Size of subsections of image for metrics
szPatch = 64;
% Laplacian kernel
lKernel = 0.25*[0,-1,0;-1,4,-1;0,-1,0];

% Collect all information for No Laplacian case
ccZ = [];
% Collect all information for With Laplacian case
ccZl = [];
% Collect cn2 values from filenames of namelist for plotting
cn_order = {};

% Pull names of all files in directory dirSims
% Filter on range, zoom to get the 8 simulated images
% Get the real image in the sharpest directory
% Run metrics

for rng = rangeV
    for zm = zoom
        
        % Import baseline and real image of first image in metric
        [dirModBase, dirReal1, basefileN, ImgNames1] = GetImageInfoMod(data_root, rng, zm);
        ImgB = double(imread(fullfile(dirModBase, basefileN)));  % Baseline image
        ImgR = double(imread(fullfile(dirReal1, ImgNames1{1}))); % Real image for comparison
        ImgR = ImgR(:,:,2);  % only green channel

        % Get the corresponding images in dir C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2
        simFiles = dir(fullfile(dirSims, '*.png'));
        SimImgNames = {simFiles(~[simFiles.isdir]).name};
        namelist = [];
        ind = 1;
        % Filter by range and zoom to get file names of range/zoom
        patt = "r" + num2str(rng) + "_z" + num2str(zm);
        for i = 1:length(SimImgNames)
            if contains(SimImgNames{:,i},patt)
                namelist{ind} = SimImgNames{:,i};
                %display(SimImgNames{:,i})
                cstr = strsplit(namelist{ind},'_c');
                cstr = strsplit(cstr{2},'.');
                cn_order = vertcat(cn_order, {rng, zm, string(cstr{1})});
                ind = ind +1;
            end
        end

        % Setup patches - Assume square images so we'll just use the image height (img_h)
        [img_h, img_w] = size(ImgB);
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
        % Identifier for patch
        index = 1;

        % Compare to 8 simulated images at same zoom/range
        for i = 1:length(namelist)
            ImgSim = double(imread(fullfile(dirSims, namelist{i})));
            
            % Create patches in baseline image and real image
            % row,col start at intv,intv
            for prow = intv:szPatch+intv:img_h-szPatch
                for pcol = intv:szPatch+intv:img_w-szPatch
                    % Baseline image  
                        % Get patch and subtract mean
                    ImgB_patch = ImgB(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    ImgB_patch = ImgB_patch - mean(ImgB_patch(:));
                        % Take Laplacian of baseline patch
                    lapImgB = conv2(ImgB_patch, lKernel, 'same');
                        %  FFT of baseline and then Laplacian
                    b_fft = fftshift(fft2(ImgB_patch)/numel(ImgB_patch));
                    lb_fft = fftshift(fft2(lapImgB)/numel(lapImgB));
                    
                    % Real Image
                        % Get patch and subtract mean
                    ImgR_patch = ImgR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    ImgR_patch = ImgR_patch - mean(ImgR_patch(:));
                        % Take Laplacian of real image patch
                    lapImgR = conv2(ImgR_patch, lKernel, 'same');
                        %  FFT of baseline and then Laplacian
                    r_fft = fftshift(fft2(ImgR_patch)/numel(ImgR_patch));
                    lr_fft = fftshift(fft2(lapImgR)/numel(lapImgR));
                    
                    % Difference between FFT of real image and baseline image
                    diff_fft_rb = r_fft - b_fft;
    
                    % Difference between FFT of Laplacian of real image and 
                    %                                   FFT of baseline image
                    diff_fft_lrb = lr_fft - lb_fft;
                    
                    % Autocorrelation/convolution:  Real and baseline image
                    cv_diff_rb = conv2(diff_fft_rb, conj(diff_fft_rb(end:-1:1, end:-1:1)), 'same');
                    s_01 = sum(abs(cv_diff_rb(:)));
                    % Autocorrelation/convolution:  Laplacian of Real and baseline image
                    cv_diff_lrb = conv2(diff_fft_lrb, conj(diff_fft_lrb(end:-1:1, end:-1:1)), 'same');
                    sl_01 = sum(abs(cv_diff_lrb(:)));            
                    
                    % Patch of simulate image i 
                    ImgSim_patch = ImgSim(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    ImgSim_patch = ImgSim_patch - mean(ImgSim_patch(:));
                    % Laplacian of simulate image i patch
                    lapImgSim = conv2(ImgSim_patch, lKernel, 'same');
                    % FFT of sim image and lap of sim image
                    sim_fft = fftshift(fft2(ImgSim_patch)/numel(ImgSim_patch));
                    lsim_fft = fftshift(fft2(lapImgSim)/numel(lapImgSim));
                    % Difference between FFTs of sim and base
                    diff_fft_simRb = sim_fft - b_fft;
                    % Difference between FFTs of Laplacians of sim and base
                    diff_fft_lsimRb = lsim_fft - lb_fft;
                    
                    % Cross correlation - Non-Laplacian case
                    cv_diff_simRb = conv2(diff_fft_rb, conj(diff_fft_simRb(end:-1:1, end:-1:1)), 'same');
                    s_02 = sum(abs(cv_diff_simRb(:)));                
                    % Cross correlation - Laplacian case
                    cv_diff_lsimRb = conv2(diff_fft_lrb, conj(diff_fft_lsimRb(end:-1:1, end:-1:1)), 'same');
                    sl_02 = sum(abs(cv_diff_lsimRb(:)));
                    % Ratios Non-Laplacian
                    r_012 = s_02/s_01;
                    r = 1-abs(1-r_012);
                    cc(index, i) = r;
                    % Ratios Laplacian
                    rl012 = sl_02/sl_01;
                    rl = 1-abs(1-rl012);
                    cc_l(index, i) = rl;
        
               end
            end
        end
        avg_r = mean(cc,1);  % non-Laplacian
        avg_rl = mean(cc_l,1); % Laplacian

        ccZ = [ccZ; rng zm numPatches*numPatches avg_r]; % non-Laplacian
        ccZl = [ccZl; rng zm numPatches*numPatches avg_rl]; % Laplacian
    end
    
end

% Plots: include real cn2/r0 values in plot
% Plot metric as a function of cn2 image (include cn2s/r0s for that range)
% Use only range/zoom combination that seem to be working

% Pairs with good metrics 
P = [600, 2500; 600, 3000; 600, 3500;600, 4000;
     700, 2000; 700, 2500; 700, 3000; 700, 3500; 700, 4000;
     800, 2000; 800, 3000; 800, 3500; 800, 4000;
     900, 4000; 1000, 5000
     ];

[numrows, numcols] = size(P);

% Plot only good pairs with Laplacian metrics
% Get cn2/r0 from csv file called turbNums for plotting

T = readtable(data_root + "modifiedBaselines\SimImgs_VaryingCn2\turbNums.csv");
% Get r0 from real image in fileA
fileA = data_root + "combined_sharpest_images_withAtmos.xlsx";
T_atmos = readtable(fileA);
for k = 1:numrows
    % Get subset of table T for cn2, r0 information
    idx = find(T.range == P(k,1));
    Tview = T(idx,:);
    Tview.strcn2 = string(Tview.cn2);
    Tview.strcn2 = strrep(Tview.strcn2,'-','');
    Tview = sortrows(Tview, "strcn2");

    % Get subset of table T_atmos to get real r0 value
    ida = find((T_atmos.range == P(k,1)) & (T_atmos.zoom == P(k,2)));
    r0_c = T_atmos{ida,"r0"};
    
    figure()
    % Filter ccZl by range/zoom - results in one row of metrics
    % Metrics were entered by namelist order
    Plotview = (ccZl(:,1) == P(k,1) & (ccZl(:,2) == P(k,2)));
    scatter(Tview.r0, ccZl(Plotview,4:end))
    grid on
    title("Laplacian Metric: Range " + num2str(P(k,1) + " Zoom " + num2str(P(k,2)) +  " with r0 of " + num2str(r0_c)))
    xlabel("Fried's Parameter r0")
    ylabel("Similarity Metric")

    fileN = fullfile(dirOut,"Lr" + num2str(P(k,1))  + "z" + num2str(P(k,2))  + ".png");

    f = gcf;
    exportgraphics(f,fileN,'Resolution',300)
end

