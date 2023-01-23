% For each range/zoom value, this script calculates similarity metrics on
% simulated images that were created by varying Cn2.
% The similarity metrics can be calculated with all of the 20 sharpest images 
% and all of the simulated images created with varying cn2/r0 values.
% The plots show the max metric and the corresponding expected value based
% on the real image at the range/zoom/Cn2 values.

clearvars
clc

% OPTIONS
savePlots = 1;
allreals = 0; % If 1, metrics will be calculated using all real images and all simulated images.
normalz = 0; % Normalize data (subtract mean and divide by std)
meanonly = 0; % This is now performed in the function turbulence_metric_noBL.m
rescaleI = 0;
rescaleMoveEnds = 0;

%rangeV = 600:50:1000;
rangeV = [600];
%zoom = [2000, 2500, 3000, 3500, 4000, 5000];
zoomV = [2000];

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
dirOut = data_root + "modifiedBaselines\NewSimulations\ByVaryingCn2\Plots\Windows\";

% Collect all information in table
TmL = table;
indT = 1;

% 1.  Find all real image file names in the sharpest directory using range/zoom values 
% 2.  Find names of all simulated files in directory dirSims by filtering
%     on range/zoom to get the simulated images created using the Python file
%     called "CreateImagesByCn2.py"
% 3.  Run metrics on each simulated image against its associated range/zoom
%     real image. 
% 4.  Enter results into table


for rng = rangeV
    % Setup table by range
    TmL = table;
    indT = 1;
    for zoom = zoomV
        display("Range " + num2str(rng) + " Zoom " + num2str(zoom))
        % Get image filenames of real images at range/zoom value
        [dirReal, ImgNames1] = GetRealImageFilenames(data_root, rng, zoom);
        if allreals == 0 % Comparing only one of the 20 sharpest images
            ImgNames1 = ImgNames1(1);
        end
        
        % Setup vector of real images vImgR
        for i = 1:length(ImgNames1) 
            % Import real image
            imgR = double(imread(fullfile(dirReal, ImgNames1{i})));
            imgR= imgR(:,:,2);  % Only green channel
            % Preprocess image
            if normalz == 1
                imgR = (imgR - mean(imgR,'all'))/std(imgR,1,'all');
            end
            if meanonly == 1
                imgR= imgR - mean(imgR,'all');
            end
            if rescaleI == 1
                imgR= rescale(imgR);
            end
            if rescaleMoveEnds == 1
                imgR= rescale(imgR) - 0.5;
            end
            vImgR{i,1} = imgR;
        end

        % Get the corresponding simulated images in the directory dirSims
        %  using the range/zoom combination with all Cn2 values
        simFiles = dir(fullfile(dirSims, '*.png'));
        SimImgNames = {simFiles(~[simFiles.isdir]).name};
        simNamelist = []; % list of all simulated image files at this zoom/range
        ind = 1;
        % Filter by range and zoom to get file names of range/zoom
        patt = "r" + num2str(rng) + "_z" + num2str(zoom);
        for i = 1:length(SimImgNames)
            if contains(SimImgNames{:,i},patt)
                simNamelist{ind,1} = SimImgNames{:,i};
                %display(namelist{ind})
                ind = ind +1;
            end
        end

        [img_h, img_w] = size(vImgR{1,1});
        removeBorder = 5;

        img_hN = img_h-2*removeBorder;
        img_wN = img_w-2*removeBorder;

        % Perform metrics by collecting metric from scanning windows
        % Size of window
        winSize = 16; %(winSize by winSize)
        pixScan = 1;
        lastPix = img_hN-winSize + 1;
        firstPix = removeBorder + 1;
         
        % Compare to simulated images to real images at same zoom/range
        for j = 1:length(vImgR)
            % Remove border
            vImgR{j,1} = vImgR{j,1}(firstPix:img_w-removeBorder,firstPix:img_w-removeBorder);
            for i = 1:length(simNamelist)
                tic;
                % cstr:  cn2 in filename (used to get r0 later)
                cstr = strsplit(simNamelist{i},'_c');
                cstr = strsplit(cstr{2},'.');
                cstr = strsplit(cstr{1},'_');
                % Read in a simulated image in namelist
                ImgSim = double(imread(fullfile(dirSims, simNamelist{i}))); % Sim Image

                if normalz == 1
                    ImgSim= (ImgSim - mean(ImgSim,'all'))/std(ImgSim,1,'all');
                end
                if meanonly == 1
                    ImgSim= ImgSim - mean(ImgSim,'all');
                end
                if rescaleI == 1
                    ImgSim= rescale(ImgSim);
                end
                if rescaleMoveEnds == 1
                    ImgSim = rescale(ImgSim) - 0.5;
                end
                ImgSim = ImgSim(firstPix:img_w-removeBorder,firstPix:img_w-removeBorder);
                   
                % Collect metrics of windows 
                if pixScan == 1
                    winMetrics = ones(lastPix,lastPix);
                end
               
                parfor col = 1:pixScan:lastPix
                    for row = 1:pixScan:lastPix
                
                        % Create windows in real image (winImgR),
                        % and simulated image(winImgSim)  
                        winImgR = vImgR{j,1}(row:row+winSize-1,col:col+winSize-1);
                        % Patch of simulated image 
                        winImgSim = ImgSim(row:row+winSize-1,col:col+winSize-1);
                        
                        m = turbulence_metric_noBL(winImgR, winImgSim);
                        winMetrics(row,col) = m;
                   end
                end
                % Calculate mean metric of all patches for this image and save to
                % table TmL
                mean_wm = mean(winMetrics,'all'); 
                std_wm = std(winMetrics,0,'all');
                TmL(indT,:) = {rng zoom string(cstr{1}) dirSims simNamelist{i} dirReal ImgNames1{j} mean_wm std_wm};
                indT = indT + 1;
                fprintf('Completed image: %s\n', fullfile(dirSims, simNamelist{i}));
                toc;
            end
        end
    end    


    %% tables
    varnames = {'Range', 'Zoom', 'Cn2str', 'SimDir','SimFilename','RealDir','RealFilename', 'MeanWindows', 'STDWindows'};
    TmL = renamevars(TmL, TmL.Properties.VariableNames, varnames);
    TmL.SimFilename = string(TmL.SimFilename);
    TmL.RealFilename = string(TmL.RealFilename);
    
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
        indG = find(TmL.Range == uniqT.Range(q) & TmL.Zoom == uniqT.Zoom(q) & TmL.Cn2str == uniqT.Cn2str(q));
        uniqT.sMetric(q) = mean(TmL.MeanWindows(indG));
        uniqT.std(q) = std(TmL.MeanWindows(indG));
        indR = find(Tr0.range == uniqT.Range(q) & Tr0.strcn2 == uniqT.Cn2str(q));
        uniqT.r0(q) = Tr0.r0(indR);
        uniqT.cn2(q) = Tr0.cn2(indR);
    end
    
    for k = 1:height(TmL)
        indK = find(Tr0.range == TmL.Range(k) & Tr0.strcn2 == TmL.Cn2str(k));
        TmL.r0(k) = Tr0.r0(indK);
        TmL.cn2(k) = Tr0.cn2(indK);
    end
    
    % Save TmL table
    writetable(TmL, dirOut + "TmL_R" + num2str(rng) + ".csv");
    
    % Sort uniqT 
    uniqT = sortrows(uniqT,["Range","Zoom","r0"]);
    writetable(uniqT, dirOut + "uniqT_R" + num2str(rng) + ".csv");


    %% Semilogx Plots
    plotcolors = ["#0072BD","#D95319","#EDB120","#7E2F8E", "#77AC30","#4DBEEE","#A2142F"];
    plots_legend = [];

    ffg = figure();
    legendL = [];
    upY = .4;
    for k = 1:length(zoomV)
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rng) & (T_atmos.zoom == zoomV(k)));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2"};
        % Setup legend entry
        txt = "Z" + num2str(zoomV(k)) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range/zoom but different Cn2 values
        indP = find(uniqT.Range == rng & uniqT.Zoom == zoomV(k));
        plots_legend(k) = semilogx(uniqT.r0(indP)*100, uniqT.sMetric(indP), '-o','Color',plotcolors(k),...
            'LineWidth',2,'MarkerSize',4);
        hold on
        
        % Collect zoom, max metric location r0
        MMetric = [uniqT.sMetric(indP) uniqT.r0(indP)];
        [max1, ind1] = max(MMetric(:,1));
        h = stem(MMetric(ind1,2)*100,1, 'MarkerFaceColor',plotcolors(k)); %,...
            %'MarkerEdgeColor',plotcolors(k)) %, 'filled')
        h.Color = plotcolors(k);
        hold on
        h2 = stem(r0_c*100,1,'MarkerFaceColor','k');
        h2.Color = plotcolors(k);
        str = "Z" + num2str(zoomV(k)) + ": Max metric " + num2str(MMetric(ind1,1)) + " at r0 " + num2str(MMetric(ind1,2)*100);
        annotation('textbox',[.68 .5 .3 upY], ...
            'String',str,'EdgeColor','none')
        upY = upY-0.05;
    end
    
    grid on
    title("Metric: Range: " + num2str(rng)) 
    legend(plots_legend,legendL, 'location', 'southeastoutside')
    xlim([min(uniqT.r0(indP)*100),max(uniqT.r0(indP)*100)])
    xlabel("Fried Parameter r_0 (cm)")
    ylabel("Mean Similarity Metric")
    x0=10;
    y0=10;
    width=900;
    ht=400;
    set(gcf,'position',[x0,y0,width,ht])
    if savePlots == 1
        f = gcf;
        fileN = fullfile(dirOut,"ALog_r" + num2str(rng) + ".png");
        %fileNf = fullfile(dirOut,"Logr" + num2str(rng) + ".fig");
        exportgraphics(f,fileN,'Resolution',300)
        %savefig(ffg,fileNf)
        %close(ffg)
    end

    %% Semilogx Plots Cn2
    plotcolors = ["#0072BD","#D95319","#EDB120","#7E2F8E", "#77AC30","#4DBEEE","#A2142F"];
    plots_legend = [];

    ffg = figure();
    legendL = [];
    upY = .4;
    for k = 1:length(zoomV) 
        % Get real image's measured cn2 and r0
        ida = find((T_atmos.range == rng) & (T_atmos.zoom == zoomV(k)));
        r0_c = T_atmos{ida,"r0"};
        cn_t = T_atmos{ida,"Cn2"};
        % Setup legend entry
        txt = "Z" + num2str(zoomV(k)) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
        legendL = [legendL; txt];
        % Find indexes in uniqT with same range/zoom but different Cn2 values
        indP = find(uniqT.Range == rng & uniqT.Zoom == zoomV(k));
        plots_legend(k) = semilogx(uniqT.cn2(indP), uniqT.sMetric(indP), '-o','Color',plotcolors(k),...
            'LineWidth',2,'MarkerSize',4);
        hold on
        
        % Collect zoom, max metric location r0
        MMetric = [uniqT.sMetric(indP) uniqT.cn2(indP)];
        [max1, ind1] = max(MMetric(:,1));
        h = stem(MMetric(ind1,2),1, 'MarkerFaceColor',plotcolors(k)); %,...
            %'MarkerEdgeColor',plotcolors(k)) %, 'filled')
        h.Color = plotcolors(k);
        hold on
        h2 = stem(cn_t,1,'MarkerFaceColor','k');
        h2.Color = plotcolors(k);
        str = "Z" + num2str(zoomV(k)) + ": Max metric " + num2str(MMetric(ind1,1)) + " at cn2 " + num2str(MMetric(ind1,2));
        annotation('textbox',[.68 .5 .3 upY], ...
            'String',str,'EdgeColor','none')
        upY = upY-0.05;
    end
    
    grid on
    title("Metric: Range: " + num2str(rng)) 
    legend(plots_legend,legendL, 'location', 'southeastoutside')
    xlim([min(uniqT.cn2(indP)),max(uniqT.cn2(indP))])
    xlabel("Cn2")
    ylabel("Mean Similarity Metric")
    x0=10;
    y0=10;
    width=900;
    ht=400;
    set(gcf,'position',[x0,y0,width,ht])
    if savePlots == 1
        f = gcf;
        fileN = fullfile(dirOut,"ALogCn2_r" + num2str(rng) + ".png");
        %fileNf = fullfile(dirOut,"Logr" + num2str(rng) + ".fig");
        exportgraphics(f,fileN,'Resolution',300)
        %savefig(ffg,fileNf)
        %close(ffg)
    end
end

