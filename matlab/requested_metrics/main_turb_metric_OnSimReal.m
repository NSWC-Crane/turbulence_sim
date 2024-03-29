% For each range/zoom value, this script calculates similarity metrics on
% images that were created to simulated Real Images.
% The similarity metrics are using one or all sharpest Real
% images as the reference and all of the 20 simulated images.  These
% simulated images were created with the new simulated image code using the
% Cn2 and Range of the real images.

clearvars
clc

% % OPTIONS
onePatch = true;  % Create only one large patch if true
savePlots = true;
allreals = false; % If true, metrics will be calculated using all real images and all simulated images.
subtractMean = true;
useLaplacian = false;

%rangeV = 600:50:1000;
rangeV = [650];
%zoom = [2000, 2500, 3000, 3500, 4000, 5000];
zoom = [2000,2500];

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
dirSims = data_root + "modifiedBaselines\NewSimulations\SimReal2\";
% Location to save plots
if onePatch == true
    dirOut = data_root + "modifiedBaselines\NewSimulations\SimReal2\Plots\OnePatch";
    patchTitle = " (One Patch)";
else
    dirOut = data_root + "modifiedBaselines\NewSimulations\SimReal2\Plots\MultiPatches";
    patchTitle = " (MultiPatches)";
end
if useLaplacian == true
 dirOut = dirOut + "\Lap";
else
 dirOut = dirOut + "\NoLap";
end

% Laplacian kernel
lKernel = 0.25*[0,-1,0;-1,4,-1;0,-1,0];

% Collect all information in table
TmL = table;
indT = 1;

% 1.  Find all real image names in the sharpest directory using range/zoom values 
% 2.  Find names of all simulated files in directory dirSims by filtering
%     on range/zoom to get the simulated images
% 3.  Run metrics

for rng = rangeV
    for zm = zoom
        display("Range " + num2str(rng) + " Zoom " + num2str(zm))
        
        [dirReal1, ImgNames1] = GetRealImageFilenames(data_root, rng, zm);
        if allreals == false
            ImgNames1 = ImgNames1(1);
        end
        
        % Setup vector of real images vImgR
        for i = 1:length(ImgNames1) 
            % Import real image
            vImgR{i,1} = double(imread(fullfile(dirReal1, ImgNames1{i}))); % Select 1st filename
            vImgR{i,1}= vImgR{i,1}(:,:,2);  % Real image for comparison - only green channel
            % Subtract Mean of Image
            if subtractMean == true
                vImgR{i,1}= vImgR{i,1} - mean(vImgR{i,1},'all');
            end
            % Find Laplacian of Image
            if useLaplacian == true
                vImgR_preLap = vImgR;
                vImgR{i,1} = conv2(vImgR{i,1}, lKernel, 'same'); % Laplacian of Real Img
                lapTitle = " Laplacian";
            else
                lapTitle = " Non-Laplacian";      
            end
        end

        % Get the corresponding simulated images using the 
        %  range/zoom combination 
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
        
        % Setup patches - Assume square images so we'll just use the image height (img_h)
        [img_h, img_w] = size(vImgR{1,1});
        % Size of subsections of image for metrics
        if onePatch == true
            numPixNot = 10;
            szPatch = floor(img_h-numPixNot);
        else
            szPatch = 62;
        end

        numPatches = floor(img_h/szPatch);
        remaining_pixels = img_h - (szPatch * numPatches);            
        if (remaining_pixels == 0)
            remaining_pixels = szPatch;
            numPatches = numPatches - 1;
        end
        
        intv = floor(remaining_pixels/(numPatches + 1));
        
        % Performing metrics
        % Compare to simulated images to real images at same zoom/range
        for j = 1:length(vImgR)
            for i = 1:length(simNamelist)
                % Read in a simulated image in namelist
                ImgSim = double(imread(fullfile(dirSims, simNamelist{i}))); % Sim Image
                if subtractMean == true
                    ImgSim= ImgSim - mean(ImgSim,'all');
                end
                if useLaplacian == true
                % Find Laplacian of Image
                    ImgSim_preLap = ImgSim;
                    ImgSim = conv2(ImgSim, lKernel, 'same'); % Laplacian of Sim Img
                end
                
                % Collect ratio 
                cc_l = [];
                % Identifier for patch
                index = 1;
                
                % Create patches in real image (ImgR_patch),
                % and simulated image(ImgSim_patch) 
                % For patches: row,col start at intv,intv
                for prow = intv:szPatch+intv:img_h-szPatch
                    for pcol = intv:szPatch+intv:img_w-szPatch
                        % Patch of real image j   
                        ImgR_patch = vImgR{j,1}(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                        % Patch of simulated image 
                        ImgSim_patch = ImgSim(prow:prow+szPatch-1,pcol:pcol+szPatch-1);

                        m = turbulence_metric_noBL(ImgR_patch, ImgSim_patch);
                        cc_l(index) = m; % Save results of all patches
                        index = index + 1;
                   end
                end
                % Calculate mean metric of all patches for this image and save to
                % table TmL
                avg_rl = mean(cc_l); % Laplacian
                std_rl = std(cc_l);
                TmL(indT,:) = {rng zm simNamelist{i} ImgNames1{j} numPatches*numPatches avg_rl std_rl}; % Laplacian
                indT = indT + 1;
            end
        end
    end    
end

%% tables
varnames = {'range', 'zoom',  'Simfilename','Realfilename','numPatches', 'simMetric', 'stdPatches'};
TmL = renamevars(TmL, TmL.Properties.VariableNames, varnames);
TmL.Simfilename = string(TmL.Simfilename);
TmL.Realfilename = string(TmL.Realfilename);

%% Create table uniqT that contains unique values of range, zoom, cn2
uniqT = unique(TmL(:,[1,2]), 'rows', 'stable');

% % Use "turbNums.csv" (created by Python file) to find r0 for plotting
% Tr0 = readtable(data_root + "modifiedBaselines\SimImgs_VaryingCn2\turbNums.csv");
% Tr0.strcn2 = string(Tr0.cn2);
% Tr0.strcn2 = strrep(Tr0.strcn2,'-','');

% Get r0 for real image in fileA - to use in plots
fileA = data_root + "combined_sharpest_images_withAtmos.xlsx";
T_atmos = readtable(fileA);
varnamesA = {'Date', 'Time', 'Time_secs', 'range', 'zoom', 'focus', 'img_filename', ...
    'img_height', 'img_width', 'pixel_step', 'start', 'stop', 'obj_size', 'Temperature', ...
    'Humidity', 'Wind_speed', 'Wind_dir', 'Bar_pressure', 'Solar_load', 'Cn2', 'r0' };
T_atmos = renamevars(T_atmos, T_atmos.Properties.VariableNames, varnamesA);

% Get mean value of similarity metric of all simulated images of same
% zoom/range/cn2 and add to table uniqT
for q = 1:height(uniqT)
    %display(q)
    indG = find(TmL.range == uniqT.range(q) & TmL.zoom == uniqT.zoom(q));
    uniqT.sMetric(q) = mean(TmL.simMetric(indG));
    uniqT.std(q) = std(TmL.simMetric(indG));
    indR = find(T_atmos.range == uniqT.range(q) & T_atmos.zoom == uniqT.zoom(q));
    uniqT.r0(q) = T_atmos.r0(indR);
    uniqT.cn2(q) = T_atmos.Cn2(indR);
end

for k = 1:height(TmL)
    indK = find(T_atmos.range == TmL.range(k) & T_atmos.zoom == TmL.zoom(k));
    TmL.r0(k) = T_atmos.r0(indK);
    TmL.cn2(k) = T_atmos.Cn2(indK);
end

% % Save TmL table
% writetable(TmL, data_root + "modifiedBaselines\SimImgs_VaryingCn2Test\TmL.csv");

% Plot by range with different colors for zoom
% Sort uniqT 
uniqT = sortrows(uniqT,["range","zoom","r0"]);


%% Plots
plotcolors = ["#0072BD","#D95319","#EDB120","#7E2F8E", "#77AC30","#4DBEEE","#A2142F",...
    "#FF0000","#00FFFF"];
plots_legend = [];
x = 1:20;
for rngP = rangeV
    ffg = figure();
    legendL = [];
    upY = .4;
    for k = 1:length(zoom) %zmP = zoom
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zoom(k)));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2"};
        % Setup legend entry
        txt = "Z" + num2str(zoom(k)) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range/zoom but different Cn2 values
        indP = find(TmL.range == rngP & TmL.zoom == zoom(k));
        plots_legend(k) = plot(x, TmL.simMetric(indP), '-o','Color',plotcolors(k),...
            'LineWidth',2,'MarkerSize',4);
        hold on
        
    end
    hold off
    grid on
    title("Range " + num2str(rngP) + ":" + lapTitle + patchTitle);
    legend(plots_legend,legendL, 'location', 'eastoutside')
    %xlim([min(uniqT.r0(indP)*100),max(uniqT.r0(indP)*100)])
    ylim([0,1.0])
    xlim([1,20])
    xlabel("Simulted Image Number")
    ylabel("Similarity Metric")
    x0=10;
    y0=10;
    width=900;
    ht=400;
    set(gcf,'position',[x0,y0,width,ht])
    if savePlots == true
        f = gcf;
        fn = "NewSim_SimulatedRealsMetrics_r" + num2str(rngP) + ".png";
        fileN = fullfile(dirOut,fn);
        exportgraphics(f,fileN,'Resolution',300)
    end
end

% Plot 2 - For each range, mean metric vs zoom value
figure()
legd = [];

ic = 1;
for rngP = rangeV
    rngstr = "" + num2str(rngP) + " ";
    legd = [legd; rngstr];
    idr = find(uniqT.range == rngP);
    plot(uniqT.zoom(idr),uniqT.sMetric(idr), '-o','Color',plotcolors(ic),...
            'LineWidth',2,'MarkerSize',4)
    xlim([2000,5000])
    ylim([0.0,1.0])
    hold on
    ic = ic + 1;
end
title("All Ranges:" + lapTitle + patchTitle);
legend(legd,'location','eastoutside')
xlabel('Zoom')
ylabel('Mean Metric')
x0=10;
y0=10;
width=700;
ht=400;
set(gcf,'position',[x0,y0,width,ht])
if savePlots == true
    f = gcf;
    fileN = fullfile(dirOut,"RealToReal_MeanMetrics.png");
    exportgraphics(f,fileN,'Resolution',300)
end