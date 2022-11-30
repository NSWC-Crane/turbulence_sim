% For each range/zoom value, this script calculates similarity metrics on files in the
% directory C:\Data\JSSAP\sharpest of real images, comparing real to real.
% The similarity metrics are calculated with one of the 20 sharpest images 
% to all of the other 20 sharpest images with the same range/zoom values.

clearvars
clc

% % OPTIONS
onePatch = false;  % Create only one large patch if true
savePlots = true;

rangeV = 600:50:1000;
%rangeV = [600,700];
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
% Location to save plots
if onePatch == true
    dirOut = data_root + "turb_metrics_RealsOnly_Plots\OnePatch";
else
    dirOut = data_root + "turb_metrics_RealsOnly_Plots\MultiPatches";
end

% Laplacian kernel
lKernel = 0.25*[0,-1,0;-1,4,-1;0,-1,0];

% Collect all information in table
TmL = table;
indT = 1;

% 1.  Find all real image names in the sharpest directory using range/zoom values 
% 2.  Run metrics

for rng = rangeV
    for zm = zoom
        display("Range " + num2str(rng) + " Zoom " + num2str(zm))
        
        [~, dirReal1, ~, ImgNames1] = GetImageInfoMod(data_root, rng, zm);
                
        % Setup vector of real images vImgR
        for i = 1:length(ImgNames1) 
            % Import real image
            vImgR{i,1} = double(imread(fullfile(dirReal1, ImgNames1{i}))); % Select 1st filename
            vImgR{i,1}= vImgR{i,1}(:,:,2);  % Real image for comparison - only green channel

            % Find Laplacian of Image
            vlapImgR{i,1} = conv2(vImgR{i,1}, lKernel, 'same'); % Laplacian of Real Img
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
              
        % Compare to one real image to all 20 real images with same zoom/range
        for j = 1:length(vlapImgR)
                
                % Collect ratio with Laplacian
                cc_l = [];
                % Identifier for patch
                index = 1;
                
                % Create patches in real image (ImgR_patch), 
                % For patches: row,col start at intv,intv
                for prow = intv:szPatch+intv:img_h-szPatch
                    for pcol = intv:szPatch+intv:img_w-szPatch
                        % Patch of real image j   
                        lapImgR_patch = vlapImgR{j,1}(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                        % Patch of reference real image  
                        lapImgRef_patch = vlapImgR{1,1}(prow:prow+szPatch-1,pcol:pcol+szPatch-1);

                        m = turbulence_metric_noBL(lapImgR_patch, lapImgRef_patch);
                        cc_l(index) = m; % Save results of all patches
                        index = index + 1;
                   end
                end
                % Calculate mean metric of all patches for this image and save to
                % table TmL
                avg_rl = mean(cc_l); % Laplacian
                std_rl = std(cc_l);
                TmL(indT,:) = {rng zm ImgNames1{1} ImgNames1{j} numPatches*numPatches avg_rl std_rl}; % Laplacian
                indT = indT + 1;
            
        end
    end    
end

%% tables
varnames = {'range', 'zoom', 'Ref_filename','Realfilename','numPatches', 'simMetric', 'stdPatches'};
TmL = renamevars(TmL, TmL.Properties.VariableNames, varnames);
TmL.Ref_filename = string(TmL.Ref_filename);
TmL.Realfilename = string(TmL.Realfilename);

%% Create table uniqT that contains unique values of range, zoom
uniqT = unique(TmL(:,[1,2]), 'rows', 'stable');

% Get r0 for real image in fileA - to use in plots
fileA = data_root + "combined_sharpest_images_withAtmos.xlsx";
T_atmos = readtable(fileA);
varnamesA = {'Date', 'Time', 'Time_secs', 'range', 'zoom', 'focus', 'img_filename', ...
    'img_height', 'img_width', 'pixel_step', 'start', 'stop', 'obj_size', 'Temperature', ...
    'Humidity', 'Wind_speed', 'Wind_dir', 'Bar_pressure', 'Solar_load', 'Cn2', 'r0' };
T_atmos = renamevars(T_atmos, T_atmos.Properties.VariableNames, varnamesA);

% Get mean value of similarity metric of all simulated images of same
% zoom/range and add to table uniqT
for q = 1:height(uniqT)
    %display(q)
    indG = find(TmL.range == uniqT.range(q) & TmL.zoom == uniqT.zoom(q) );
    uniqT.sMetric(q) = mean(TmL.simMetric(indG));
    uniqT.std(q) = std(TmL.simMetric(indG));
    % Add cn2 and r0 from atmos.xlsx
    indAtm = find(T_atmos.range == uniqT.range(q) & T_atmos.zoom == uniqT.zoom(q));
    uniqT.cn2(q) = T_atmos.Cn2(indAtm);
    uniqT.r0(q) = T_atmos.r0(indAtm);
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
uniqT = sortrows(uniqT,["range","zoom"]);


%% Plots
% 1.  For each range, plot image number vs metric for each zoom value (use TmL)
% 2.  For each range, plot zoom value vs mean metric (use uniqT)
plotcolors = ["#0072BD","#D95319","#EDB120","#7E2F8E", "#77AC30","#4DBEEE","#A2142F",...
    "#FF0000","#00FFFF"];
% Plot 1 
for rngP = rangeV
    figure()
    legd = [];
    for i = 1:length(zoom)
        indP = find((TmL.range == rngP) & (TmL.zoom == zoom(i)));
        x = 1:20;
        plot(x,TmL.simMetric(indP), '-o','Color',plotcolors(i),...
            'LineWidth',2,'MarkerSize',4)
        legd = [legd; ""+ num2str(zoom(i)+ " ")];
        hold on
    end
    hold off
    title("Range " + num2str(rngP));
    xlim([1,20])
    ylim([0.55,1.0])
    xlabel("Image Number")
    ylabel("Metric")
    legend(legd, 'location', 'eastoutside')
    x0=10;
    y0=10;
    width=700;
    ht=400;
    set(gcf,'position',[x0,y0,width,ht])
    if savePlots == true
        f = gcf;
        fileN = fullfile(dirOut,"RealToReal_Range" + num2str(rngP) + ".png");
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
    ylim([0.6,1.0])
    hold on
    ic = ic + 1;
end
legend(legd,'location','eastoutside')
xlabel('Zoom')
ylabel('Metric')
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

