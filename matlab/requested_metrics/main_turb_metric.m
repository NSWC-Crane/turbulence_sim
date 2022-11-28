% For each range/zoom value, this script calculates similarity metrics on files in the
% directory C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2.
% The similarity metrics are calculated with all of the 20 sharpest images 
% and all of the simulated images created with varying cn2/r0 values

% Note:
% 1. Calculates Laplacian before creating patches
% 2. Does not use baseline image
% 3. Does not subtract mean of image
% 4. Averages metrics using all simulated images and all 20 sharpest real, 
%    if allreals is set to true.

% To Do:
% 1.  Add annotation/text to show max metric for each zoom value on plot.
% 2.  Add column to tables TmL and uniqT to store standard deviation of patches and in metrics
% overall
% 3.  Change units of Fried Parameter to centimeters
% 4.  Add abilty to use simulated image as the reference image.

clearvars
clc

% % OPTIONS
onePatch = false;  % Create only one large patch if true
savePlots = true;
allreals = true; % If true, metrics will be calculated using all real images and all simulated images.

%rangeV = 600:50:1000;
rangeV = [650];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];
%zoom = [2500];

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
dirOut = data_root + "modifiedBaselines\SimImgs_VaryingCn2\Plots5";

% Laplacian kernel
lKernel = 0.25*[0,-1,0;-1,4,-1;0,-1,0];

% Collect all information in table
TmL = table;
indT = 1;

% 1.  Find all real image names in the sharpest directory using range/zoom values 
% 2.  Find names of all simulated files in directory dirSims by filtering
%     on range/zoom to get the simulated images created using the Python file
%     called "CreateImagesByCn2.py"
% 3.  Run metrics

for rng = rangeV
    for zm = zoom
        display("Range " + num2str(rng) + " Zoom " + num2str(zm))
        
        [~, dirReal1, ~, ImgNames1] = GetImageInfoMod(data_root, rng, zm);
        if allreals == false
            ImgNames1 = ImgNames1(1);
        end
        
        % Setup vector of real images vImgR
        for i = 1:length(ImgNames1) 
            % Import real image
            vImgR{i,1} = double(imread(fullfile(dirReal1, ImgNames1{i}))); % Select 1st filename
            vImgR{i,1}= vImgR{i,1}(:,:,2);  % Real image for comparison - only green channel

            % Find Laplacian of Image
            vlapImgR{i,1} = conv2(vImgR{i,1}, lKernel, 'same'); % Laplacian of Real Img
        end

        % Get the corresponding simulated images in the directory called
        %  C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2 using the 
        %  range/zoom combination with all Cn2 values
        simFiles = dir(fullfile(dirSims, '*.png'));
        SimImgNames = {simFiles(~[simFiles.isdir]).name};
        simNamelist = []; % list of all simulated image files at this zoom/range
        ind = 1;
        % Filter by range and zoom to get file names of range/zoom
        patt = "r" + num2str(rng) + "_z" + num2str(zm);
        for i = 1:length(SimImgNames)
            if contains(SimImgNames{:,i},patt)
                simNamelist{ind,1} = SimImgNames{:,i};
                %display(namelist{ind})
                ind = ind +1;
            end
        end
        
        % Performing metrics
        % Setup patches - Assume square images so we'll just use the image height (img_h)
        [img_h, img_w] = size(vImgR{1,1});
        % Size of subsections of image for metrics
        if onePatch == true
            numPixNot = 10;
            szPatch = floor(img_h-numPixNot);
            strPtch = "_OnePtch";
            titlePtch = " (One Patch)";
        else
            szPatch = 62;
            strPtch = "";
            titlePtch = "";
        end

        numPatches = floor(img_h/szPatch);
        remaining_pixels = img_h - (szPatch * numPatches);            
        if (remaining_pixels == 0)
            remaining_pixels = szPatch;
            numPatches = numPatches - 1;
        end
        
        intv = floor(remaining_pixels/(numPatches + 1));
              
        % Compare to simulated images to real images at same zoom/range
        for j = 1:length(vImgR)
            for i = 1:length(simNamelist)
                % Read in a simulated image in namelist
                % cstr:  cn2 in filename (used to get r0 later)
                cstr = strsplit(simNamelist{i},'_c');
                cstr = strsplit(cstr{2},'.');
                cstr = strsplit(cstr{1},'_');
    
                ImgSim = double(imread(fullfile(dirSims, simNamelist{i}))); % Sim Image
                
                % Find Laplacian of image
                lapImgSim = conv2(ImgSim, lKernel, 'same');  % Laplacian of Sim Image
    
                % Collect ratio with Laplacian
                cc_l = [];
                % Identifier for patch
                index = 1;
                
                % Create patches in real image (ImgR_patch),
                % and simulated image(ImgSim_patch) 
                % For patches: row,col start at intv,intv
                for prow = intv:szPatch+intv:img_h-szPatch
                    for pcol = intv:szPatch+intv:img_w-szPatch
                        % Patch of real image j   
                        lapImgR_patch = vlapImgR{j,1}(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                        % Patch of simulated image 
                        lapImgSim_patch = lapImgSim(prow:prow+szPatch-1,pcol:pcol+szPatch-1);

                        m = turbulence_metric_noBL(lapImgR_patch, lapImgSim_patch);
                        cc_l(index) = m; % Save results of all patches
                        index = index + 1;
                   end
                end
                % Calculate mean metric of all patches for this image and save to
                % table TmL
                avg_rl = mean(cc_l); % Laplacian
                std_rl = std(cc_l);
                TmL(indT,:) = {rng zm string(cstr{1}) simNamelist{i} ImgNames1{j} numPatches*numPatches avg_rl std_rl}; % Laplacian
                indT = indT + 1;
            end
        end
    end    
end

%% tables
varnames = {'range', 'zoom', 'cn2str', 'Simfilename','Realfilename','numPatches', 'simMetric', 'stdPatches'};
TmL = renamevars(TmL, TmL.Properties.VariableNames, varnames);
TmL.filename = string(TmL.Simfilename);
TmL.filename = string(TmL.Realfilename);

%% Create table uniqT that contains unique values of range, zoom, cn2
uniqT = unique(TmL(:,[1,2,3]), 'rows', 'stable');

% Use "turbNums.csv" (created by Python file) to find r0 for plotting
Tr0 = readtable(data_root + "modifiedBaselines\SimImgs_VaryingCn2\turbNums.csv");
Tr0.strcn2 = string(Tr0.cn2);
Tr0.strcn2 = strrep(Tr0.strcn2,'-','');

% Get mean value of similarity metric of all simulated images of same
% zoom/range/cn2 and add to table uniqT
for q = 1:height(uniqT)
    %display(q)
    indG = find(TmL.range == uniqT.range(q) & TmL.zoom == uniqT.zoom(q) & TmL.cn2str == uniqT.cn2str(q));
    uniqT.sMetric(q) = mean(TmL.simMetric(indG));
    uniqT.std(q) = std(TmL.simMetric(indG));
    indR = find(Tr0.range == uniqT.range(q) & Tr0.strcn2 == uniqT.cn2str(q));
    uniqT.r0(q) = Tr0.r0(indR);
    uniqT.cn2(q) = Tr0.cn2(indR);
end

for k = 1:height(TmL)
    indK = find(Tr0.range == TmL.range(k) & Tr0.strcn2 == TmL.cn2str(k));
    TmL.r0(k) = Tr0.r0(indK);
    TmL.cn2(k) = Tr0.cn2(indK);
end

% % Save TmL table
% writetable(TmL, data_root + "modifiedBaselines\SimImgs_VaryingCn2Test\TmL.csv");

% Get r0 for real image in fileA - to use in plots
fileA = data_root + "combined_sharpest_images_withAtmos.xlsx";
T_atmos = readtable(fileA);

% Plot by range with different colors for zoom
% Sort uniqT 
uniqT = sortrows(uniqT,["range","zoom","r0"]);

%% Plots
% for rngP = rangeV
%     ffg = figure();
%     legendL = [];
%     for zmP = zoom
%         % Get the real image's measured cn2 value and calculated r0
%         ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zmP));
%         r0_c = T_atmos{ida,"r0"};
%         cn_t = T_atmos{ida,"Cn2_m___2_3_"};
%         % Setup legend entry
%         txt = "Z" + num2str(zmP) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
%         legendL = [legendL; txt];
%         % Find indexes in uniqT with same range and zoom but different Cn2
%         % values
%         indP = find(uniqT.range == rngP & uniqT.zoom == zmP);
%         plot(uniqT.r0(indP)*100, uniqT.sMetric(indP), '-o',...
%             'LineWidth',2,...
%             'MarkerSize',3)
%         hold on
%     end
%     grid on 
%     title("Laplacian Metric: Range: " + num2str(rngP) + titlePtch)
%     legend(legendL, 'location', 'northeastoutside')
%     xlim([min(uniqT.r0(indP)*100),max(uniqT.r0(indP)*100)])
%     xlabel("Fried Parameter r_0 (cm)")
%     ylabel("Mean Similarity Metric")
%     x0=10;
%     y0=10;
%     width=900;
%     ht=400;
%     set(gcf,'position',[x0,y0,width,ht]) 
%     if savePlots == true
%         f = gcf;
%         fileN = fullfile(dirOut, "Lr" + num2str(rngP) + strPtch + ".png");
%         fileNf = fullfile(dirOut, "Lr" + num2str(rngP) + strPtch + ".fig");
%         exportgraphics(f,fileN,'Resolution',300)
%         savefig(ffg,fileNf)
%         close(ffg)
%     end
% end

%% Semilogx Plots
for rngP = rangeV
    ffg = figure();
    legendL = [];
    upY = .4;
    for zmP = zoom
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zmP));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2_m___2_3_"};
        % Setup legend entry
        txt = "Z" + num2str(zmP) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range/zoom but different Cn2 values
        indP = find(uniqT.range == rngP & uniqT.zoom == zmP);
        semilogx(uniqT.r0(indP)*100, uniqT.sMetric(indP), '-o',...
            'LineWidth',2,...
            'MarkerSize',4)
         hold on
        % Collect zoom, max metric location r0
        MMetric = [uniqT.sMetric(indP) uniqT.r0(indP)];
        [max1, ind1] = max(MMetric(:,1));
        str = "Z" + num2str(zmP) + ": Max metric " + num2str(MMetric(ind1,1)) + " at r0 " + num2str(MMetric(ind1,2)*100);
        annotation('textbox',[.68 .5 .3 upY], ...
            'String',str,'EdgeColor','none')
        upY = upY-0.05;
    end
    
    grid on
    title("Laplacian Metric: Range: " + num2str(rngP) + titlePtch) 
    legend(legendL, 'location', 'southeastoutside')
    xlim([min(uniqT.r0(indP)*100),max(uniqT.r0(indP)*100)])
    xlabel("Fried Parameter r_0 (cm)")
    ylabel("Mean Similarity Metric")
    x0=10;
    y0=10;
    width=900;
    ht=400;
    set(gcf,'position',[x0,y0,width,ht])
    if savePlots == true
        f = gcf;
        fileN = fullfile(dirOut,"LogLr" + num2str(rngP) + strPtch + ".png");
        fileNf = fullfile(dirOut,"LogLr" + num2str(rngP) + strPtch + ".fig");
        exportgraphics(f,fileN,'Resolution',300)
        savefig(ffg,fileNf)
        close(ffg)
    end
end

%% Plots for Standard Deviation Information
% % Plot standard deviation of patches used to calculate metric of each image
% % Sort TmL by Cn2 value and range/zoom - Each Patch
% TmL = sortrows(TmL,["range","zoom","r0"]);
% figure()
% x = 1:height(TmL);
% plot(x,TmL.simMetric, '-o',...
%             'LineWidth',1,...
%             'MarkerSize',2)
% hold on
% plot(x,TmL.stdPatches, '-o',...
%             'LineWidth',1,...
%             'MarkerSize',2)
% grid on
% xlabel("Image Number")
% ylabel ("Mean Metric and Std")
% legend('Mean','Standard Deviation','location', 'east')
% title("Mean and Std of Patches for Each Image")
% fileS = fullfile(dirOut,"Std_Patches.png");
% x0=10;
% y0=10;
% width=800;
% ht=400;
% set(gcf,'position',[x0,y0,width,ht])
% if savePlots == true
%     f = gcf;
%     exportgraphics(f,fileS,'Resolution',300)
% end
% 
% % Plot standard deviation of all simulated images used to calculate metric
% % for each range/zoom/cn2 value.
% % Metric over 20 simulated images
% % Sorted uniqT earlier:  uniqT = sortrows(uniqT,["range","zoom","r0"]);
% figure()
% x = 1:height(uniqT);
% plot(x, uniqT.sMetric, '-o',...
%             'LineWidth',1,...
%             'MarkerSize',2)
% hold on
% plot(x, uniqT.std, '-o',...
%             'LineWidth',1,...
%             'MarkerSize',2)
% grid on
% xlabel("Set Number")
% ylabel ("Mean and Std of Metric")
% legend('Mean','Standard Deviation','location', 'east')
% title("Mean and Std of 20 Simulated Image Sets")
% fileS = fullfile(dirOut,"Std_SimImages.png");
% x0=10;
% y0=10;
% width=800;
% ht=400;
% set(gcf,'position',[x0,y0,width,ht])
% if savePlots == true
%     f = gcf;
%     exportgraphics(f,fileS,'Resolution',300)
% end
% 