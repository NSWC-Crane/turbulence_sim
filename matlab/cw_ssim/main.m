% Test CW-SSIM index on Ranges/Zoom values
% Function cwssim_index.m:
% CW-SSIM Index, Version 1.0
% Copyright(c) 2010  Zhou Wang, Mehul Sampat and Alan Bovik
% All Rights Reserved.

% Real images are the Reference Images - green channel only
% Test Images:  Comparing similarity of Reference Images to New Simulated Images
% Test reference images to simulated images with varying Cn2 values
% Does not subtract mean (normalize images).

clearvars
clc

% OPTIONS
savePlots = true;

%rangeV = 600:50:1000;
rangeV = [650, 700, 750];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];
%zoom = [2000, 2500, 3000];

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
dirSims = data_root + "modifiedBaselines\NewSimulations\ByVaryingCn2\";
% Location to save plots
dirOut = data_root + "modifiedBaselines\NewSimulations\ByVaryingCn2\cwssimPlots\";

% Collect all information in table
TmL = table;
indT = 1;

% Constants for cwssim_index
% Example:  cwssim = cwssim_index(img1, img2,6,16,0,0);
level = 2;
or = 10; % 10
guardb = 0;
K = 0;

% 1.  Find all real image file names in the sharpest directory using range/zoom values 
% 2.  Find names of all simulated files in directory dirSims by filtering
%     on range/zoom to get the simulated images created using the Python file
%     called "CreateImagesByCn2.py"
% 3.  Run metrics on each simulated image against its associated range/zoom
%     real image. 
% 4.  Enter results into table

for rng = rangeV
    for zm = zoom
        display("Range " + num2str(rng) + " Zoom " + num2str(zm))

        % Real image as reference image
        [dirReal1, ImgNames1] = GetRealImageFilenames(data_root, rng, zm);
        % Import real image as reference image.
        RefImg = double(imread(fullfile(dirReal1, ImgNames1{1})));
        RefImg = RefImg(:,:,2); % Green channel only
        % Remove 5 pixels off borders
        RefImg = RefImg(6:end-5, 6:end-5);

        % Get the corresponding simulated images in the directory dirSims
        %  using the range/zoom combination with all Cn2 values
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
        
        % Perform metrics    
        % Compare to simulated images to real images at same zoom/range
        for i = 1:length(simNamelist)
            % cstr:  cn2 in filename (used to get r0 later)
            cstr = strsplit(simNamelist{i},'_c');
            cstr = strsplit(cstr{2},'.');
            cstr = strsplit(cstr{1},'_');
            % Read in a simulated image in namelist
            ImgSim = double(imread(fullfile(dirSims, simNamelist{i}))); % Sim Image
            % Remove 5 pixels off borders
            ImgSim = ImgSim(6:end-5, 6:end-5);
            ImgSim_gauss = imgaussfilt(ImgSim,.69);

            % Find CW-SSIM Index
            cwssim = cwssim_index(RefImg, ImgSim, level, or, guardb, K);
            cwssimG = cwssim_index(RefImg, ImgSim_gauss, level, or, guardb, K);


            TmL(indT,:) = {rng zm string(cstr{1}) simNamelist{i} ImgNames1{1} cwssim, cwssimG};
            indT = indT + 1;
        end

    end    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% tables
varnames = {'range', 'zoom', 'cn2str', 'Simfilename','Realfilename', 'cwssim', 'cwssimG'};
TmL = renamevars(TmL, TmL.Properties.VariableNames, varnames);
TmL.Simfilename = string(TmL.Simfilename);
TmL.Realfilename = string(TmL.Realfilename);

%% Create table uniqT that contains unique values of range, zoom, cn2
uniqT = unique(TmL(:,[1,2,3]), 'rows', 'stable');

% Use "turbNums.csv" (created by Python file) to find r0 for plotting
Tr0 = readtable(data_root + "modifiedBaselines\SimImgs_VaryingCn2\turbNums.csv");
Tr0.strcn2 = string(Tr0.cn2);
Tr0.strcn2 = strrep(Tr0.strcn2,'-','');

% Real image data in fileA - to use in plots
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
    indG = find(TmL.range == uniqT.range(q) & TmL.zoom == uniqT.zoom(q) & TmL.cn2str == uniqT.cn2str(q));
    uniqT.cwssim(q) = mean(TmL.cwssim(indG));
    uniqT.cwssimG(q) = mean(TmL.cwssimG(indG));
     uniqT.std(q) = std(TmL.cwssim(indG));
     uniqT.stdG(q) = std(TmL.cwssimG(indG));
    indR = find(Tr0.range == uniqT.range(q) & Tr0.strcn2 == uniqT.cn2str(q));
    uniqT.r0(q) = Tr0.r0(indR);
    uniqT.cn2(q) = Tr0.cn2(indR);
end

%  Add r0 and Cn2 to TmL table
for k = 1:height(TmL)
    indK = find(Tr0.range == TmL.range(k) & Tr0.strcn2 == TmL.cn2str(k));
    TmL.r0(k) = Tr0.r0(indK);
    TmL.cn2(k) = Tr0.cn2(indK);
end

% % Save TmL table
% writetable(TmL, data_root + "modifiedBaselines\SimImgs_VaryingCn2Test\TmL.csv");

% Sort uniqT 
uniqT = sortrows(uniqT,["range","zoom","cn2"]);

%% Semilogx Plots
plotcolors = ["#0072BD","#D95319","#EDB120","#7E2F8E", "#77AC30","#4DBEEE","#A2142F"];
plots_legend = [];
for rngP = rangeV
    ffg = figure();
    legendL = [];
    upY = .4;
    for k = 1:length(zoom) 
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zoom(k)));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2"};
        % Setup legend entry
        txt = "Z" + num2str(zoom(k)) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range/zoom but different Cn2 values
        indP = find(uniqT.range == rngP & uniqT.zoom == zoom(k));
        plots_legend(k) = semilogx(uniqT.r0(indP)*100, uniqT.cwssim(indP), '-o','Color',plotcolors(k),...
            'LineWidth',2,'MarkerSize',4);
        hold on
        
        % Collect zoom, max metric location r0
        MMetric = [uniqT.cwssim(indP) uniqT.r0(indP) uniqT.cn2(indP)];
        [max1, ind1] = max(MMetric(:,1));
        h = stem(MMetric(ind1,2)*100,1, 'MarkerFaceColor',plotcolors(k)); %,...
            %'MarkerEdgeColor',plotcolors(k)) %, 'filled')
        h.Color = plotcolors(k);
        hold on
        h2 = stem(r0_c*100,1,'MarkerFaceColor','k');
        h2.Color = plotcolors(k);
        str = "Z" + num2str(zoom(k)) + ": Max metric " + num2str(MMetric(ind1,1)) + ...
            " at r0 " + num2str(MMetric(ind1,2)*100) + " cn2 " + num2str(MMetric(ind1,3));
        annotation('textbox',[.732 .5 .3 upY], ...
            'String',str,'EdgeColor','none')
        upY = upY-0.05;
    end
    
    grid on
    title("CW-SSIM Index (No Blur): Range " + num2str(rngP)) 
    legend(plots_legend,legendL, 'location', 'southeastoutside')
    xlim([min(uniqT.r0(indP)*100),max(uniqT.r0(indP)*100)])
    xlabel("Fried Parameter r_0 (cm)")
    ylabel("CW-SSIM")
    x0=10;
    y0=10;
    width=1200;
    ht=600;
    set(gcf,'position',[x0,y0,width,ht])
    if savePlots == true
        f = gcf;
        fileN = fullfile(dirOut,"r0_NoBlur_Log_r" + num2str(rngP) + ".png");
        %fileNf = fullfile(dirOut,"Logr" + num2str(rngP) + strPtch + ".fig");
        exportgraphics(f,fileN,'Resolution',300)
        %savefig(ffg,fileNf)
        %close(ffg)
    end
end

% Cn2
plots_legend = [];
for rngP = rangeV
    ffg = figure();
    legendL = [];
    upY = .4;
    for k = 1:length(zoom) 
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zoom(k)));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2"};
        % Setup legend entry
        txt = "Z" + num2str(zoom(k)) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range/zoom but different Cn2 values
        indP = find(uniqT.range == rngP & uniqT.zoom == zoom(k));
        plots_legend(k) = semilogx(uniqT.cn2(indP), uniqT.cwssim(indP), '-o','Color',plotcolors(k),...
            'LineWidth',2,'MarkerSize',4);
        hold on
        
        % Collect zoom, max metric location r0
        MMetric = [uniqT.cwssim(indP) uniqT.r0(indP) uniqT.cn2(indP)];
        [max1, ind1] = max(MMetric(:,1));
        h = stem(MMetric(ind1,3),1, 'MarkerFaceColor',plotcolors(k)); %,...
            %'MarkerEdgeColor',plotcolors(k)) %, 'filled')
        h.Color = plotcolors(k);
        hold on
        % Plot real cn2
        h2 = stem(cn_t,1,'MarkerFaceColor','k');
        h2.Color = plotcolors(k);
        str = "Z" + num2str(zoom(k)) + ": Max metric " + num2str(MMetric(ind1,1)) + ...
            " at r0 " + num2str(MMetric(ind1,2)*100) + " cn2 " + num2str(MMetric(ind1,3));
        annotation('textbox',[.732 .5 .3 upY], ...
            'String',str,'EdgeColor','none')
        upY = upY-0.05;
    end
    
    grid on
    title("CW-SSIM Index (No Blur): Range " + num2str(rngP)) 
    legend(plots_legend,legendL, 'location', 'southeastoutside')
    xlim([min(uniqT.cn2(indP)),max(uniqT.cn2(indP))])
    xlabel("Cn2")
    ylabel("CW-SSIM")
    x0=10;
    y0=10;
    width=1200;
    ht=600;
    set(gcf,'position',[x0,y0,width,ht])
    if savePlots == true
        f = gcf;
        fileN = fullfile(dirOut,"cn2_NoBlur_Log_r" + num2str(rngP) + ".png");
        %fileNf = fullfile(dirOut,"Logr" + num2str(rngP) + strPtch + ".fig");
        exportgraphics(f,fileN,'Resolution',300)
        %savefig(ffg,fileNf)
        %close(ffg)
    end
end

%% cwssimG

plotcolors = ["#0072BD","#D95319","#EDB120","#7E2F8E", "#77AC30","#4DBEEE","#A2142F"];
plots_legend = [];
for rngP = rangeV
    ffg = figure();
    legendL = [];
    upY = .4;
    for k = 1:length(zoom) 
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zoom(k)));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2"};
        % Setup legend entry
        txt = "Z" + num2str(zoom(k)) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range/zoom but different Cn2 values
        indP = find(uniqT.range == rngP & uniqT.zoom == zoom(k));
        plots_legend(k) = semilogx(uniqT.r0(indP)*100, uniqT.cwssimG(indP), '-o','Color',plotcolors(k),...
            'LineWidth',2,'MarkerSize',4);
        hold on
        
        % Collect zoom, max metric location r0
        MMetric = [uniqT.cwssimG(indP) uniqT.r0(indP) uniqT.cn2(indP)];
        [max1, ind1] = max(MMetric(:,1));
        h = stem(MMetric(ind1,2)*100,1, 'MarkerFaceColor',plotcolors(k)); %,...
            %'MarkerEdgeColor',plotcolors(k)) %, 'filled')
        h.Color = plotcolors(k);
        hold on
        h2 = stem(r0_c*100,1,'MarkerFaceColor','k');
        h2.Color = plotcolors(k);
        str = "Z" + num2str(zoom(k)) + ": Max metric " + num2str(MMetric(ind1,1)) + ...
            " at r0 " + num2str(MMetric(ind1,2)*100) + " cn2 " + num2str(MMetric(ind1,3));
        annotation('textbox',[.732 .5 .3 upY], ...
            'String',str,'EdgeColor','none')
        upY = upY-0.05;
    end
    
    grid on
    title("CW-SSIM Index (Blur): Range " + num2str(rngP)) 
    legend(plots_legend,legendL, 'location', 'southeastoutside')
    xlim([min(uniqT.r0(indP)*100),max(uniqT.r0(indP)*100)])
    xlabel("Fried Parameter r_0 (cm)")
    ylabel("CW-SSIM")
    x0=10;
    y0=10;
    width=1200;
    ht=600;
    set(gcf,'position',[x0,y0,width,ht])
    if savePlots == true
        f = gcf;
        fileN = fullfile(dirOut,"r0_Blur_Log_r" + num2str(rngP) + ".png");
        %fileNf = fullfile(dirOut,"Logr" + num2str(rngP) + strPtch + ".fig");
        exportgraphics(f,fileN,'Resolution',300)
        %savefig(ffg,fileNf)
        %close(ffg)
    end
end

% Cn2
plots_legend = [];
for rngP = rangeV
    ffg = figure();
    legendL = [];
    upY = .4;
    for k = 1:length(zoom) 
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zoom(k)));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2"};
        % Setup legend entry
        txt = "Z" + num2str(zoom(k)) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range/zoom but different Cn2 values
        indP = find(uniqT.range == rngP & uniqT.zoom == zoom(k));
        plots_legend(k) = semilogx(uniqT.cn2(indP), uniqT.cwssimG(indP), '-o','Color',plotcolors(k),...
            'LineWidth',2,'MarkerSize',4);
        hold on
        
        % Collect zoom, max metric location r0
        MMetric = [uniqT.cwssimG(indP) uniqT.r0(indP) uniqT.cn2(indP)];
        [max1, ind1] = max(MMetric(:,1));
        h = stem(MMetric(ind1,3),1, 'MarkerFaceColor',plotcolors(k)); %,...
            %'MarkerEdgeColor',plotcolors(k)) %, 'filled')
        h.Color = plotcolors(k);
        hold on
        h2 = stem(cn_t,1,'MarkerFaceColor','k');
        h2.Color = plotcolors(k);
        str = "Z" + num2str(zoom(k)) + ": Max metric " + num2str(MMetric(ind1,1)) + ...
            " at r0 " + num2str(MMetric(ind1,2)*100) + " cn2 " + num2str(MMetric(ind1,3));
        annotation('textbox',[.732 .5 .3 upY], ...
            'String',str,'EdgeColor','none')
        upY = upY-0.05;
    end
    
    grid on
    title("CW-SSIM Index (Blur): Range " + num2str(rngP)) 
    legend(plots_legend,legendL, 'location', 'southeastoutside')
    xlim([min(uniqT.cn2(indP)),max(uniqT.cn2(indP))])
    xlabel("Cn2")
    ylabel("CW-SSIM")
    x0=10;
    y0=10;
    width=1200;
    ht=600;
    set(gcf,'position',[x0,y0,width,ht])
    if savePlots == true
        f = gcf;
        fileN = fullfile(dirOut,"cn2_Blur_Log_r" + num2str(rngP) + ".png");
        %fileNf = fullfile(dirOut,"Logr" + num2str(rngP) + strPtch + ".fig");
        exportgraphics(f,fileN,'Resolution',300)
        %savefig(ffg,fileNf)
        %close(ffg)
    end
end


