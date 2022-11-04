% For each range, zoom value perform similarity metrics on files in the
% directory C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2
% Similarity metrics between one of the 20 sharpest images and all
% simulated images created with varying cn2/r0 values

clearvars
clc

% rangeV = 600:50:1000;
rangeV = [600];
% zoom = [2000, 2500, 3000, 3500, 4000, 5000];
zoom = [ 2000];

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
dirOut = data_root + "modifiedBaselines\SimImgs_VaryingCn2\Plots2";

% Size of subsections of image for metrics
szPatch = 148;
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
        ImgB = double(imread(fullfile(dirModBase, basefileN)));  % Baseline image
        ImgR = double(imread(fullfile(dirReal1, ImgNames1{1}))); % Real image for comparison
        ImgR = ImgR(:,:,2);  % only green channel

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
            ImgSim = double(imread(fullfile(dirSims, namelist{i})));
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
                        % Get patch and subtract mean ℱ
                    ImgB_patch = ImgB(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    ImgB_patch = ImgB_patch - mean(ImgB_patch(:));
                        % Take Laplacian of baseline patch
                    lapImgB = conv2(ImgB_patch, lKernel, 'same');
                        %  FFT of baseline and then Laplacian
%                     b_fft = fftshift(fft2(ImgB_patch)/numel(ImgB_patch));
                    lb_fft = fftshift(fft2(lapImgB)/numel(lapImgB));
                    
                    % Real Image ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_0−𝜇_0 ))
                        % Get patch and subtract mean 
                    ImgR_patch = ImgR(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    ImgR_patch = ImgR_patch - mean(ImgR_patch(:));
                        % Take Laplacian of real image patch
                    lapImgR = conv2(ImgR_patch, lKernel, 'same');
                        %  FFT of baseline and then Laplacian
%                     r_fft = fftshift(fft2(ImgR_patch)/numel(ImgR_patch));
                    lr_fft = fftshift(fft2(lapImgR)/numel(lapImgR));
                    
                    % Difference between FFT of real image and baseline image
%                     diff_fft_rb = r_fft - b_fft;
    
                    % Difference between FFT of Laplacian of real image and 
                    %                                   FFT of baseline image
                    %  EQN 1 Final:  〖𝐼𝑚𝑔〗_0𝑏=ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_0−𝜇_0 ))−ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_𝑏−𝜇_𝑏 ))?
%                     diff_fft_lrb = lr_fft - lb_fft;
                    diff_fft_lrb = lr_fft;
                    
                    % Autocorrelation/convolution:  Real and baseline image
%                     cv_diff_rb = conv2(diff_fft_rb, conj(diff_fft_rb(end:-1:1, end:-1:1)), 'same');
%                     s_01 = sum(abs(cv_diff_rb(:)));
                    % Autocorrelation/convolution:  Laplacian of Real and baseline image
                    % EQN 3: 𝒦_00=𝐸[〖𝐼𝑚𝑔〗_0𝑏∙(〖𝐼𝑚𝑔〗_0𝑏 ) ̅ ]
                    cv_diff_lrb = conv2(diff_fft_lrb, conj(diff_fft_lrb(end:-1:1, end:-1:1)), 'same');
                    sl_01 = sum(abs(cv_diff_lrb(:)));            
                    
                    % Patch of simulate image i 
                    % EQN 2:  Setup
                    ImgSim_patch = ImgSim(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                    ImgSim_patch = ImgSim_patch - mean(ImgSim_patch(:));
                    % Laplacian of simulate image i patch
                    lapImgSim = conv2(ImgSim_patch, lKernel, 'same');
                    % FFT of sim image and lap of sim image
%                     sim_fft = fftshift(fft2(ImgSim_patch)/numel(ImgSim_patch));
                    lsim_fft = fftshift(fft2(lapImgSim)/numel(lapImgSim));
                    % Difference between FFTs of sim and base
%                     diff_fft_simRb = sim_fft - b_fft;
                    % EQN 2:  Final 〖𝐼𝑚𝑔〗_1𝑏=ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_1−𝜇_1 ))−ℱ(𝛻^2 (〖𝐼𝑚𝑔〗_𝑏−𝜇_𝑏 ))
                    % Difference between FFTs of Laplacians of sim and base
%                     diff_fft_lsimRb = lsim_fft - lb_fft;
                    diff_fft_lsimRb = lsim_fft;
                    
                    % Cross correlation - Non-Laplacian case
                    % EQN 4:  𝒦_01=𝐸[〖𝐼𝑚𝑔〗_0𝑏∙(〖𝐼𝑚𝑔〗_1𝑏 ) ̅ ]
%                     cv_diff_simRb = conv2(diff_fft_rb, conj(diff_fft_simRb(end:-1:1, end:-1:1)), 'same');
%                     s_02 = sum(abs(cv_diff_simRb(:)));                
                    % Cross correlation - Laplacian case
                    cv_diff_lsimRb = conv2(diff_fft_lrb, conj(diff_fft_lsimRb(end:-1:1, end:-1:1)), 'same');
                    sl_02 = sum(abs(cv_diff_lsimRb(:)));
                    % Ratios Non-Laplacian
%                     r_012 = s_02/s_01;
%                     r = 1-abs(1-r_012);  % Non-Laplacian metric



%                     cc(index) = r;

                    % Ratios Laplacian
                    % EQN 5 ℳ_01=1−|1−(∑2_0^(𝑗−1)▒𝒦_01 )/(∑2_0^(𝑗−1)▒𝒦_00 )| 
                    %      where 𝑗 is the number of pixels in the image
                    rl012 = sl_02/sl_01;
                    rl = 1-abs(1-rl012);  % Laplacian metric
                    cc_l(index) = rl;
                   
                    if(false)
                        
                        figure(99);
                        subplot(1,3,1)
                        surf(ImgB_patch);
                        shading interp;
                        subplot(1,3,2)
                        surf(ImgR_patch);
                        shading interp;
                        subplot(1,3,3)
                        surf(ImgSim_patch);
                        shading interp;
                        colormap(jet(256));                        
                        
                        
                        figure(100);
                        subplot(1,3,1)
                        surf(lapImgB);
                        shading interp;
                        subplot(1,3,2)
                        surf(lapImgR);
                        shading interp;
                        subplot(1,3,3)
                        surf(lapImgSim);
                        shading interp;
                        colormap(jet(256));

                        figure(101);
                        subplot(1,3,1)
                        surf(abs(lb_fft));
                        shading interp;
                        subplot(1,3,2)
                        surf(abs(lr_fft));
                        shading interp;
                        subplot(1,3,3)
                        surf(abs(lsim_fft));
                        shading interp;
                        colormap(jet(256));

                        figure(102);
                        subplot(1,2,1)
                        surf(abs(diff_fft_lrb));
                        shading interp;
                        subplot(1,2,2)
                        surf(abs(diff_fft_lsimRb));
                        shading interp;
                        colormap(jet(256));

                        figure(103);
                        subplot(1,2,1)
                        surf(abs(cv_diff_lrb));
                        shading interp;
                        subplot(1,2,2)
                        surf(abs(cv_diff_lsimRb));
                        shading interp;
                        colormap(jet(256));                    
                    end
                    
                    index = index + 1;
        
               end
            end
            % Calculate mean metric of all patches for this image and save to
            % tables Tm and TmL
%             avg_r = mean(cc);  % non-Laplacian
            avg_rl = mean(cc_l); % Laplacian
    
%             Tm(indT,:) = {rng zm string(cstr{1}) namelist{i} numPatches*numPatches avg_r}; % non-Laplacian
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
Tr0 = readtable(data_root + "modifiedBaselines\SimImgs_VaryingCn2\turbNums_600.csv");
Tr0.strcn2 = string(Tr0.cn2);
Tr0.strcn2 = strrep(Tr0.strcn2,'-','');

% Get mean value of similarity metric of all simulated images of same
% zoom/range and add to table uniqT
for q = 1:height(uniqT)
    indG = find(TmL.range == uniqT.range(q) & TmL.zoom == uniqT.zoom(q) & TmL.cn2str == uniqT.cn2str(q));
    uniqT.sMetric(q) = mean(TmL.simMetric(indG));
    indR = find(Tr0.range == uniqT.range(q) & Tr0.strcn2 == uniqT.cn2str(q));
    uniqT.r0(q) = Tr0.r0(indR);
    uniqT.cn2(q) = Tr0.cn2(indR);
end

% % Save ccZl and ccZ
% writematrix(ccZl, data_root + "modifiedBaselines\SimImgs_VaryingCn2\ccZl.csv")
% writematrix(ccZ, data_root + "modifiedBaselines\SimImgs_VaryingCn2\ccZ.csv")

% % Pairs with good metrics - determined when compared only 1 simulated image
% P = [600, 2500; 600, 3000; 600, 3500;600, 4000;
%      700, 2000; 700, 2500; 700, 3000; 700, 3500; 700, 4000;
%      800, 2000; 800, 3000; 800, 3500; 800, 4000;
%      900, 4000; 1000, 5000
%      ];
% 
% [numrows, numcols] = size(P);
%  
% Get r0 for real image in fileA
fileA = data_root + "combined_sharpest_images_withAtmos.xlsx";
T_atmos = readtable(fileA);

% Plot by range with different colors for zoom
% Sort uniqT 
uniqT = sortrows(uniqT,["range","zoom","r0"]);

%% Create all plots
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
%         % Determine if this zoom/range combination had good metrics and
%         % mark the plot
%         idf = find((P(:,1) == rngP) & (P(:,2)) == zmP);
%         if isempty(idf)
%             str = '\leftarrow B';
%         else
%             str = '\leftarrow G';
%         end
%         xf = uniqT.r0(indP(16));
%         yf = uniqT.sMetric(indP(16));
%         text(xf, yf, str)

        hold on
    end
    grid on
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

    fileN = fullfile(dirOut,"Lr" + num2str(rngP)  + ".png");
    f = gcf;
    exportgraphics(f,fileN,'Resolution',300)

end

%% Plot against log(r0) Semi Log Plot
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
%         % Determine if this zoom/range combination had good metrics and
%         % mark the plot
%         idf = find((P(:,1) == rngP) & (P(:,2)) == zmP);
%         if isempty(idf)
%             str = '\leftarrow B';
%         else
%             str = '\leftarrow G';
%         end
%         xf = uniqT.r0(indP(15));
%         yf = uniqT.sMetric(indP(15));
%         text(xf, yf, str)
        hold on
    end
    grid on
    title("Laplacian Metric: Range: " + num2str(rngP))
    legend(legendL, 'location', 'northeastoutside')
    xlim([min(uniqT.r0(indP)),max(uniqT.r0(indP))])
    xlabel("Fried's Parameter r_0")
    ylabel("Mean Similarity Metric M_0_1")
    x0=10;
    y0=10;
    width=900;
    height=400;
    set(gcf,'position',[x0,y0,width,height])

    fileN = fullfile(dirOut,"LogLr" + num2str(rngP)  + ".png");
    f = gcf;
    exportgraphics(f,fileN,'Resolution',300)

end

