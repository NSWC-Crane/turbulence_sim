% For each range, zoom value perform similarity metrics on files in the
% directory C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2
% Similarity metrics between one of the 20 sharpest images and all
% simulated images created with varying cn2/r0 values

% Changes:
% 1.  Calculates Laplacian before creating patches
% 2.  Adds option to subtract mean of image

clearvars
clc

onePatch = false;
subtractMean = true;

%rangeV = 600:50:1000;
rangeV = [600];
%zoom = [2000, 2500, 3000, 3500, 4000, 5000];
zoom = [2000, 2500]; %, 3000];

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
dirSims = data_root + "modifiedBaselines\SimImgs_VaryingCn2Test";
% Location to save plots
dirOut = data_root + "modifiedBaselines\SimImgs_VaryingCn2Test\Plots";

% Laplacian kernel
lKernel = 0.25*[0,-1,0;-1,4,-1;0,-1,0];

% Collect all information for No Laplacian case
Tm = table;
% Collect all information for With Laplacian case
TmL = table;
indT = 1;

% Pull names of all files in directory dirSims
% Filter on range, zoom to get the simulated images created using Python
% file called "CreateImagesByCn2.py"
% Get the real image using range/zoom values in the sharpest directory
% Run metrics

for rng = rangeV
    for zm = zoom
        display("Range " + num2str(rng) + " Zoom " + num2str(zm))
        % Import baseline and real image (first of 20 real images in
        % sharpest directory).
        % Simulated images are compared to baseline and real image by
        % zoom/range values.
        [dirModBase, dirReal1, basefileN, ImgNames1] = GetImageInfoMod(data_root, rng, zm);
        % Read in images: Baseline, Real
        ImgB = double(imread(fullfile(dirModBase, basefileN)));  % Baseline image
        ImgR = double(imread(fullfile(dirReal1, ImgNames1{1}))); 
        ImgR = ImgR(:,:,2);  % Real image for comparison - only green channel
        
        %Subtract mean of images if option is set
        if subtractMean == true
            ImgB = ImgB - mean(ImgB(:));
            ImgR = ImgR - mean(ImgR(:));
        end

        % Find Laplacian of Images
        lapImgB = conv2(ImgB, lKernel, 'same'); % Laplacian of Baseline Image
        lapImgR = conv2(ImgR, lKernel, 'same'); % Laplacian of Real Img

        % Get the corresponding simulated images in 
        % the directory C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2.
        simFiles = dir(fullfile(dirSims, '*.png'));
        SimImgNames = {simFiles(~[simFiles.isdir]).name};
        namelist = []; % list of all simulated image files at this zoom/range
        ind = 1;
        % Filter by range and zoom to get file names of range/zoom
        patt = "r" + num2str(rng) + "_z" + num2str(zm);
        for i = 1:length(SimImgNames)
            if contains(SimImgNames{:,i},patt)
                namelist{ind} = SimImgNames{:,i};
                %display(namelist{ind})
                ind = ind +1;
            end
        end

        % Setup patches - Assume square images so we'll just use the image height (img_h)
        [img_h, img_w] = size(ImgB);
        % Size of subsections of image for metrics
        if onePatch == true
            numPixNot = 10;
            szPatch = floor(img_h-numPixNot);
        else
            szPatch = 64;
        end

        numPatches = floor(img_h/szPatch);
        remaining_pixels = img_h - (szPatch * numPatches);            
        if (remaining_pixels == 0)
            remaining_pixels = szPatch;
            numPatches = numPatches - 1;
        end
        
        intv = floor(remaining_pixels/(numPatches + 1));
      
        % Compare to simulated images at same zoom/range
        for i = 1:length(namelist)
            % Read in a simulated image in namelist
            % cstr:  cn2 in filename (used to get r0 later)
            cstr = strsplit(namelist{i},'_c');
            cstr = strsplit(cstr{2},'.');
            cstr = strsplit(cstr{1},'_');
            ImgSim = double(imread(fullfile(dirSims, namelist{i}))); % Sim Image
            %Subtract mean of images if option is set
            if subtractMean == true
                ImgSim = ImgSim - mean(ImgSim(:));
            end
            % Find Laplacian of image
            lapImgSim = conv2(ImgSim, lKernel, 'same');  % Laplacian of Sim Image

            % Collect ratio without Laplacian
            cc = [];
            % Collect ratio with Laplacian
            cc_l = [];
            % Identifier for patch
            index = 1;
            
            % Create patches in baseline image (ImgB_patch), real image (ImgR_patch),
            % and simulated image(ImgSim_patch) 
            % For patches: row,col start at intv,intv
            for prow = intv:szPatch+intv:img_h-szPatch
                for pcol = intv:szPatch+intv:img_w-szPatch
                    % EQN 1: Setup
                    % Baseline image (𝛻^2 (〖𝐼𝑚𝑔〗_𝑏−𝜇_𝑏 ))  
                    % Define patch of Baseline Image
                    ImgB_patch = ImgB(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    % Define patch of Laplacian of Baseline Image
                    lapImgB_patch = lapImgB(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    %  FFT of patches: baseline and Laplacian
                    b_fft = fftshift(fft2(ImgB_patch)/numel(ImgB_patch));
                    lb_fft = fftshift(fft2(lapImgB_patch)/numel(lapImgB_patch));
                    
                    % Real Image ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_0−𝜇_0 ))
                    % Define patch of Real Image
                    ImgR_patch = ImgR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    lapImgR_patch = lapImgR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    %  FFT of real patches:  real and Laplacian
                    r_fft = fftshift(fft2(ImgR_patch)/numel(ImgR_patch));
                    lr_fft = fftshift(fft2(lapImgR_patch)/numel(lapImgR_patch));
                    
                    % Difference between FFT of real image and baseline
                    % image patches
                    diff_fft_rb = r_fft - b_fft;
    
                    % Difference between FFT of Laplacian of real image and 
                    %                       FFT of baseline image patches
                    %  EQN 1 Final:  〖𝐼𝑚𝑔〗_0𝑏=ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_0−𝜇_0 ))−ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_𝑏−𝜇_𝑏 ))?
                    diff_fft_lrb = lr_fft - lb_fft;
                    
                    % Autocorrelation/convolution:  Real and baseline image
                    cv_diff_rb = conv2(diff_fft_rb, conj(diff_fft_rb(end:-1:1, end:-1:1)), 'same');
                    s_01 = sum(abs(cv_diff_rb(:)));
                    % Autocorrelation/convolution:  Laplacian of Real and baseline image
                    % EQN 3: 𝒦_00=𝐸[〖𝐼𝑚𝑔〗_0𝑏∙(〖𝐼𝑚𝑔〗_0𝑏 ) ̅ ]
                    cv_diff_lrb = conv2(diff_fft_lrb, conj(diff_fft_lrb(end:-1:1, end:-1:1)), 'same');
                    sl_01 = sum(abs(cv_diff_lrb(:)));            
                    
                    % Patch of simulated image i 
                    % EQN 2:  Setup
                    ImgSim_patch = ImgSim(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    lapImgSim_patch = lapImgSim(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    % FFT of sim image and lap of sim image patches
                    sim_fft = fftshift(fft2(ImgSim_patch)/numel(ImgSim_patch));
                    lsim_fft = fftshift(fft2(lapImgSim_patch)/numel(lapImgSim_patch));
                    % Difference between FFTs of sim and base
                    diff_fft_simRb = sim_fft - b_fft;
                    % EQN 2:  Final 〖𝐼𝑚𝑔〗_1𝑏=ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_1−𝜇_1 ))−ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_𝑏−𝜇_𝑏 ))
                    % Difference between FFTs of Laplacians of sim and base
                    diff_fft_lsimRb = lsim_fft - lb_fft;
                    
                    % Cross correlation - Non-Laplacian case                    
                    cv_diff_simRb = conv2(diff_fft_rb, conj(diff_fft_simRb(end:-1:1, end:-1:1)), 'same');
                    s_02 = sum(abs(cv_diff_simRb(:)));
                    % Cross correlation - Laplacian case
                    % EQN 4:  𝒦_01=𝐸[〖𝐼𝑚𝑔〗_0𝑏∙(〖𝐼𝑚𝑔〗_1𝑏 ) ̅ ]
                    cv_diff_lsimRb = conv2(diff_fft_lrb, conj(diff_fft_lsimRb(end:-1:1, end:-1:1)), 'same');
                    sl_02 = sum(abs(cv_diff_lsimRb(:)));
                    % Ratios Non-Laplacian
                    r_012 = s_02/s_01;
                    r = 1-abs(1-r_012);  % Non-Laplacian metric
                    cc(index) = r;
                    % Ratios Laplacian
                    % EQN 5 ℳ_01=1−|1−(∑2_0^(𝑗−1)▒𝒦_01 )/(∑2_0^(𝑗−1)▒𝒦_00 )| 
                    %      where 𝑗 is the number of pixels in the image
                    rl012 = sl_02/sl_01;
                    rl = 1-abs(1-rl012);  % Laplacian metric
                    cc_l(index) = rl;

                    index = index + 1;
        
               end
            end
            % Calculate mean metric of all patches for this image and save to
            % tables Tm and TmL
            avg_r = mean(cc);  % non-Laplacian
            avg_rl = mean(cc_l); % Laplacian
    
            Tm(indT,:) = {rng zm string(cstr{1}) namelist{i} numPatches*numPatches avg_r}; % non-Laplacian
            TmL(indT,:) = {rng zm string(cstr{1}) namelist{i} numPatches*numPatches avg_rl}; % Laplacian
            indT = indT + 1;
        end
        
    end
    
end

varnames = {'range', 'zoom', 'cn2str', 'filename','numPatches', 'simMetric'};
TmL = renamevars(TmL, TmL.Properties.VariableNames, varnames);
TmL.filename = string(TmL.filename);

% Create table uniqT that contains unique values of range, zoom, cn2
uniqT = unique(TmL(:,[1,2,3]), 'rows', 'stable');

% Use "trubNums.csv" (created by Python file) to find r0 for plotting
Tr0 = readtable(data_root + "modifiedBaselines\SimImgs_VaryingCn2\turbNums.csv");
Tr0.strcn2 = string(Tr0.cn2);
Tr0.strcn2 = strrep(Tr0.strcn2,'-','');

% Get mean value of similarity metric of all simulated images of same
% zoom/range/cn2 and add to table uniqT
for q = 1:height(uniqT)
    indG = find(TmL.range == uniqT.range(q) & TmL.zoom == uniqT.zoom(q) & TmL.cn2str == uniqT.cn2str(q));
    uniqT.sMetric(q) = mean(TmL.simMetric(indG));
    indR = find(Tr0.range == uniqT.range(q) & Tr0.strcn2 == uniqT.cn2str(q));
    uniqT.r0(q) = Tr0.r0(indR);
    uniqT.cn2(q) = Tr0.cn2(indR);
end

% % Save Tm and TmL
% writetable(Tm, data_root + "modifiedBaselines\SimImgs_VaryingCn2Test\Tm.csv");
% writetable(TmL, data_root + "modifiedBaselines\SimImgs_VaryingCn2Test\TmL.csv");

% Get r0 for real image in fileA
fileA = data_root + "combined_sharpest_images_withAtmos.xlsx";
T_atmos = readtable(fileA);

% Plot by range with different colors for zoom
% Sort uniqT 
uniqT = sortrows(uniqT,["range","zoom","r0"]);
% writetable(uniqT, data_root + "modifiedBaselines\SimImgs_VaryingCn2Test\uniqT.csv");

% Create all plots
for rngP = rangeV
    figure()
    legendL = [];
    for zmP = zoom
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zmP));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2_m___2_3_"};
        % Setup legend entry
        txt = "Z" + num2str(zmP) + " r0 " + num2str(r0_c) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range and zoom but different Cn2
        % values
        indP = find(uniqT.range == rngP & uniqT.zoom == zmP);
        plot(uniqT.r0(indP), uniqT.sMetric(indP), '-o',...
            'LineWidth',2,...
            'MarkerSize',3)
        hold on
    end
    grid on
    if subtractMean == true
        fileN = fullfile(dirOut,"SubMean_Lr" + num2str(rngP)  + ".png");
        title("Laplacian Metric: Range: " + num2str(rngP) + " - Subtracted Mean")
    else
        fileN = fullfile(dirOut,"NoSubMean_Lr" + num2str(rngP)  + ".png");
        title("Laplacian Metric: Range: " + num2str(rngP) + " - Did not subtract mean")
    end
    title("Laplacian Metric: Range " + num2str(rngP))
    legend(legendL, 'location', 'northeastoutside')
    xlim([min(uniqT.r0(indP)),max(uniqT.r0(indP))])
    xlabel("Fried's Parameter r_0")
    ylabel("Mean Similarity Metric M_0_1")
    x0=10;
    y0=10;
    width=900;
    height=400;
    set(gcf,'position',[x0,y0,width,height])
    
%     if subtractMean == true
%         fileN = fullfile(dirOut,"SubMean_Lr" + num2str(rngP)  + ".png");
%     else
%         fileN = fullfile(dirOut,"NoSubMean_Lr" + num2str(rngP)  + ".png");
%     end
    f = gcf;
    exportgraphics(f,fileN,'Resolution',300)

end

% Plot against log(r0) Semi Log Plot
for rngP = rangeV
    figure()
    legendL = [];
    for zmP = zoom
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zmP));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2_m___2_3_"};
        % Setup legend entry
        txt = "Z" + num2str(zmP) + " r0 " + num2str(r0_c) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range and zoom but different Cn2
        % values
        indP = find(uniqT.range == rngP & uniqT.zoom == zmP);
        semilogx(uniqT.r0(indP), uniqT.sMetric(indP), '-o',...
            'LineWidth',2,...
            'MarkerSize',4)
        hold on
    end
    grid on
    if subtractMean == true
        fileN = fullfile(dirOut,"SubMean_LogLr" + num2str(rngP)  + ".png");
        title("Laplacian Metric: Range: " + num2str(rngP) + " - Subtracted Mean")
    else
        fileN = fullfile(dirOut,"NoSubMean_LogLr" + num2str(rngP)  + ".png");
        title("Laplacian Metric: Range: " + num2str(rngP) + " - Did not subtract mean")
    end
    
    legend(legendL, 'location', 'northeastoutside')
    xlim([min(uniqT.r0(indP)),max(uniqT.r0(indP))])
    xlabel("Fried's Parameter r_0")
    ylabel("Mean Similarity Metric M_0_1")
    x0=10;
    y0=10;
    width=900;
    height=400;
    set(gcf,'position',[x0,y0,width,height])

    
    f = gcf;
    exportgraphics(f,fileN,'Resolution',300)

end

