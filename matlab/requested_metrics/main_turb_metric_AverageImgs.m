% For each range/zoom value, this script calculates similarity metrics on
% simulated images that were created by varying Cn2.
% In this variation, all 20 sharpest, real images are averaged (before and
% after alignment), creating the reference images.  The test image is the average
% of all 20 simulated images at a specific Cn2 value.  Then metrics are calculated
% between the reference average image and the test average images to find
% the Cn2 that aligns best with the reference average image.

clearvars
clc

% OPTIONS
savePlots = true;

%rangeV = 600:50:1000;
rangeV = [600, 650, 700, 750, 800];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];
%zoom = [3500];
cn2Values = [7e-16, 8e-16, 9e-16, ...
            1e-15, 2e-15, 3e-15, 4e-15, 5e-15, 6e-15, 7e-15, 8e-15, 9e-15,...
            1e-14, 2e-14, 3e-14, 4e-14, 5e-14, 6e-14, 7e-14, 8e-14, 9e-14,...
            1e-13, 2e-13, 3e-13, 4e-13, 5e-13, 6e-13, 7e-13, 8e-13, 9e-13,...
            1e-12];
cn2strs = ["7e16", "8e16", "9e16", ...
            "1e15", "2e15", "3e15", "4e15", "5e15", "6e15", "7e15", "8e15", "9e15",...
            "1e14", "2e14", "3e14", "4e14", "5e14", "6e14", "7e14", "8e14", "9e14",...
            "1e13", "2e13", "3e13", "4e13", "5e13", "6e13", "7e13", "8e13", "9e13",...
            "1e12"];

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
dirOut = data_root + "modifiedBaselines\NewSimulations\ByVaryingCn2\Plots\AvgImgs\";

% Collect all information in table
Tm = table;
indT = 1;

% 1.  Find all real image file names in the sharpest directory using range/zoom values 
% 2.  Use 20 sharpest, real images at range/zoom to create average
% reference image.  Save image.
% 3.  Find names of all simulated files in directory dirSims by filtering
%     on range/zoom to get the simulated images created using the new
%     simulation code.
% 4.  Create average test image from these 20 simulated images.  Save
% image.
% 5.  Run metrics on each average simulated image against its associated range/zoom
%     reference average image. 
% 6.  Enter results into table

% Notes:
% Subtract mean to normalize data.
% Trim borders of each image (5 pixels).
% No longer using patches.

for rng = rangeV
    for zm = zoom
        display("Range " + num2str(rng) + " Zoom " + num2str(zm))
        
        [dirReal, RealImgFNames] = GetRealImageFilenames(data_root, rng, zm);
        totalRefImg = 0;        
        % Average the 20 sharpest images 
        for i = 1:length(RealImgFNames) 
            % Import real image
            RImg = double(imread(fullfile(dirReal, RealImgFNames{i})));
            % Use only green channel
            RImg= RImg(:,:,2); 
            % Trim image by removing pixels on borders
            RImg = RImg(6:end-5, 6:end-5);
            % Normalize image - subtract mean of image
            RImg = RImg - mean(RImg,'all');
            totalRefImg = totalRefImg + RImg; 
        end
        % Calcuate average image
        AvgRefImg = totalRefImg./length(RealImgFNames);
        fName = "avgReal_r" + num2str(rng) + "_z" + num2str(zm) + ".png";
        imwrite(uint8(AvgRefImg), fullfile(dirOut, fName))

        % Get the corresponding simulated images in the directory dirSims
        %  using the range/zoom combination with all Cn2 values
        simFiles = dir(fullfile(dirSims, '*.png'));
        SimImgNames = {simFiles(~[simFiles.isdir]).name};
        

        % Group simulated images by range, zoom, and Cn2 value
        % Average images
        % Calculate metric
        % Store resutls in table

        for indC = 1:length(cn2Values)
            simNamelist = []; % list of all simulated image files at this zoom/range/cn2
            totalSimImg = 0;
            % Filter by range, zoom, cn2
            patt = "r" + num2str(rng) + "_z" + num2str(zm) + "_c" + num2str(cn2strs(indC));
            ind = 1;
            for i = 1:length(SimImgNames)
                if contains(SimImgNames{:,i},patt)
                    simNamelist{ind,1} = SimImgNames{:,i};
                    SimImg = double(imread(fullfile(dirSims, simNamelist{ind})));
                    % Trim image by removing pixels on borders
                    SimImg = SimImg(6:end-5, 6:end-5);
                    % Normalize image - subtract mean of image
                    SimImg= SimImg - mean(SimImg,'all');
                    totalSimImg = totalSimImg + SimImg;
                    ind = ind + 1;
                end
            end
            % Average 20 simulated images
            AvgSimImg = totalSimImg./length(simNamelist);
            fName = "avgSim_r" + num2str(rng) + "_z" + num2str(zm) + "_c" + cn2strs(indC) + ".png";
            imwrite(uint8(AvgSimImg), fullfile(dirOut, fName))

            % Calculate metric and add to table Tm
            m = turbulence_metric_noBL(AvgRefImg, AvgSimImg);
            Tm(indT,:) = {rng zm cn2Values(indC) simNamelist{1} dirReal RealImgFNames{1} m};
            indT = indT + 1;

        end
    end    
end

%% Tables
varnames = {'range', 'zoom', 'Cn2', 'SimFilename','RealDirectory','RealFilename', 'simMetric'};
Tm = renamevars(Tm, Tm.Properties.VariableNames, varnames);
Tm.SimFilename = string(Tm.SimFilename);
Tm.RealFilename = string(Tm.RealFilename);

% Use "turbNums.csv" (created by Python file) to find r0 for plotting
Tr0 = readtable(data_root + "modifiedBaselines\SimImgs_VaryingCn2\turbNums.csv");

% Real image data in fileA - to use in plots
fileA = data_root + "combined_sharpest_images_withAtmos.xlsx";
T_atmos = readtable(fileA);
varnamesA = {'Date', 'Time', 'Time_secs', 'range', 'zoom', 'focus', 'img_filename', ...
    'img_height', 'img_width', 'pixel_step', 'start', 'stop', 'obj_size', 'Temperature', ...
    'Humidity', 'Wind_speed', 'Wind_dir', 'Bar_pressure', 'Solar_load', 'Cn2', 'r0' };
T_atmos = renamevars(T_atmos, T_atmos.Properties.VariableNames, varnamesA);

for k = 1:height(Tm)
    indK = find(Tr0.range == Tm.range(k) & Tr0.cn2 == Tm.Cn2(k));
    Tm.r0(k) = Tr0.r0(indK);
end

% % Save Tm table
% writetable(Tm, data_root + "modifiedBaselines\SimImgs_VaryingCn2Test\Tm.csv");

% Sort rows 
Tm = sortrows(Tm,["range","zoom","Cn2"]);

% %% Semilogx Plots
% plotcolors = ["#0072BD","#D95319","#EDB120","#7E2F8E", "#77AC30","#4DBEEE","#A2142F"];
% plots_legend = [];
% for rngP = rangeV
%     ffg = figure();
%     legendL = [];
%     upY = .4;
%     for k = 1:length(zoom) %zmP = zoom
%         % Get real image's measured cn2 and r0
%         ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zoom(k)));
%         r0_c = T_atmos{ida,"r0"};
%         cn_t = T_atmos{ida,"Cn2"};
%         % Setup legend entry
%         txt = "Z" + num2str(zoom(k)) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
%         legendL = [legendL; txt];
%         % Find indexes in Tm with same range/zoom but different Cn2 values
%         indP = find(Tm.range == rngP & Tm.zoom == zoom(k));
%         plots_legend(k) = semilogx(Tm.r0(indP)*100, Tm.simMetric(indP), '-o','Color',plotcolors(k),...
%             'LineWidth',2,'MarkerSize',4);
%         hold on
%         
%         % Collect zoom, max metric location r0
%         MMetric = [Tm.simMetric(indP) Tm.r0(indP) Tm.Cn2(indP)];
%         [max1, ind1] = max(MMetric(:,1));
%         h = stem(MMetric(ind1,2)*100,1, 'MarkerFaceColor',plotcolors(k)); %,...
%             %'MarkerEdgeColor',plotcolors(k)) %, 'filled')
%         h.Color = plotcolors(k);
%         hold on
%         h2 = stem(r0_c*100,1,'MarkerFaceColor','k');
%         h2.Color = plotcolors(k);
%         str = "Z" + num2str(zoom(k)) + ": Max metric " + num2str(MMetric(ind1,1)) + ...
%             " at r0 " + num2str(MMetric(ind1,2)*100) + " cn2 " + num2str(MMetric(ind1,3));
%         annotation('textbox',[.715 .5 .3 upY], ...
%             'String',str,'EdgeColor','none')
%         upY = upY-0.05;
%     end
%     
%     grid on
%     title("Metric of Averages: Range " + num2str(rngP)) 
%     legend(plots_legend,legendL, 'location', 'southeastoutside')
%     xlim([min(Tm.r0(indP)*100),max(Tm.r0(indP)*100)])
%     xlabel("Fried Parameter r_0 (cm)")
%     ylabel("Similarity Metric")
%     x0=10;
%     y0=10;
%     width=1100;
%     ht=500;
%     set(gcf,'position',[x0,y0,width,ht])
%     if savePlots == true
%         f = gcf;
%         fileN = fullfile(dirOut,"Log_r" + num2str(rngP) + ".png");
%         %fileNf = fullfile(dirOut,"Logr" + num2str(rngP) + ".fig");
%         exportgraphics(f,fileN,'Resolution',300)
%         %savefig(ffg,fileNf)
%         %close(ffg)
%     end
% end
% 
% % Plot Cn2 on x-axis
% plots_legend = [];
% for rngP = rangeV
%     ffg = figure();
%     legendL = [];
%     upY = .4;
%     for k = 1:length(zoom) %zmP = zoom
%         % Get real image's measured cn2 and r0
%         ida = find((T_atmos.range == rngP) & (T_atmos.zoom == zoom(k)));
%         r0_c = T_atmos{ida,"r0"};
%         cn_t = T_atmos{ida,"Cn2"};
%         % Setup legend entry
%         txt = "Z" + num2str(zoom(k)) + " r0 " + num2str(r0_c*100) + " Cn2 " + num2str(cn_t);
%         legendL = [legendL; txt];
%         % Find indexes in Tm with same range/zoom but different Cn2 values
%         indP = find(Tm.range == rngP & Tm.zoom == zoom(k));
%         plots_legend(k) = semilogx(Tm.Cn2(indP), Tm.simMetric(indP), '-o','Color',plotcolors(k),...
%             'LineWidth',2,'MarkerSize',4);
%         hold on
%         
%         % Collect zoom, max metric location r0
%         MMetric = [Tm.simMetric(indP) Tm.r0(indP) Tm.Cn2(indP)];
%         [max1, ind1] = max(MMetric(:,1));
%         h = stem(MMetric(ind1,3),1, 'MarkerFaceColor',plotcolors(k)); %,...
%             %'MarkerEdgeColor',plotcolors(k)) %, 'filled')
%         h.Color = plotcolors(k);
%         hold on
%         h2 = stem(cn_t,1,'MarkerFaceColor','k');
%         h2.Color = plotcolors(k);
%         str = "Z" + num2str(zoom(k)) + ": Max metric " + num2str(MMetric(ind1,1)) + ...
%             " at r0 " + num2str(MMetric(ind1,2)*100) + " cn2 " + num2str(MMetric(ind1,3));
%         annotation('textbox',[.715 .5 .3 upY], ...
%             'String',str,'EdgeColor','none')
%         upY = upY-0.05;
%     end
%     
%     grid on
%     title("Metric of Averages: Range " + num2str(rngP)) 
%     legend(plots_legend,legendL, 'location', 'southeastoutside')
%     xlim([min(Tm.Cn2(indP)),max(Tm.Cn2(indP))])
%     xlabel("Cn2")
%     ylabel("Similarity Metric")
%     x0=10;
%     y0=10;
%     width=1100;
%     ht=500;
%     set(gcf,'position',[x0,y0,width,ht])
%     if savePlots == true
%         f = gcf;
%         fileN = fullfile(dirOut,"cn2_Log_r" + num2str(rngP) + ".png");
%         %fileNf = fullfile(dirOut,"cn2_Logr" + num2str(rngP) + ".fig");
%         exportgraphics(f,fileN,'Resolution',300)
%         %savefig(ffg,fileNf)
%         %close(ffg)
%     end
% end
