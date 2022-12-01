% Compares simulated image at same range/zoom/cn2 to one of the simulated
% images.

clearvars
clc

% % OPTIONS
onePatch = true;  % Create only one large patch if true
savePlots = true;

%rangeV = 600:50:1000;
rangeV = [650,700,800,900];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];
%zoom = [5000];
% cn2str = ["c1e12", "c1e13","c1e14","c1e15",...
%             "c2e12", "c2e13","c2e14","c2e15",...
%             "c3e12", "c3e13","c3e14","c3e15",...
%             "c4e12", "c4e13","c4e14","c4e15",...
%             "c5e12", "c5e13","c5e14","c5e15","c5e16",...
%             "c6e12", "c6e13","c6e14","c6e15","c6e16",...
%             "c7e12", "c7e13","c7e14","c7e15","c7e16",...
%             "c8e12", "c8e13","c8e14","c8e15","c8e16",...
%             "c9e12", "c9e13","c9e14","c9e15","c9e16"
%             ];
cn2str = ["c1e14","c1e15",...
           "c2e14","c2e15",...
           "c3e14","c3e15",...
            "c4e14","c4e15",...
            "c5e13","c5e14","c5e15",...
            "c6e13","c6e14","c6e15",...
            "c7e13","c7e14","c7e15",...
            "c8e13","c8e14","c8e15","c8e16",...
            "c9e13","c9e14","c9e15","c9e16"
            ];

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
dirOut = data_root + "modifiedBaselines\SimImgs_SameCn2\Plots2";

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
        for cns = cn2str
            display("Range " + num2str(rng) + " Zoom " + num2str(zm) + " Cn2 " + cns)
            
            % Get the corresponding simulated images in the directory called
            %  C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2 using the 
            %  range/zoom combination with all Cn2 values
            simFiles = dir(fullfile(dirSims, '*.png'));
            SimImgNames = {simFiles(~[simFiles.isdir]).name};
            simNamelist = []; % list of all simulated image files at this zoom/range
            ind = 1;
            % Filter by range and zoom to get file names of range/zoom
            patt = "r" + num2str(rng) + "_z" + num2str(zm) + "_" + cns;
            for i = 1:length(SimImgNames)
                if contains(SimImgNames{:,i},patt)
                    simNamelist{ind,1} = SimImgNames{:,i};
                    %display(namelist{ind})
                    ind = ind +1;
                end
            end
    
            % Read in reference simulated image
            ImgSimRef = double(imread(fullfile(dirSims, simNamelist{1,1})));
            % Find Laplacian of image
            lapImgSimRef = conv2(ImgSimRef, lKernel, 'same');  % Laplacian of Sim Image
    
            % Performing metrics
            % Setup patches - Assume square images so we'll just use the image height (img_h)
            [img_h, img_w] = size(ImgSimRef);  
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
                  
            % Compare simulated images to one reference simulated image at same
            % zoom/range/cn2
            for i = 1:length(simNamelist)
                % Read in a simulated image in namelist
                % cstr:  cn2 in filename (used to get r0 later)
                cstr = strsplit(simNamelist{i},'_c');
                cstr = strsplit(cstr{2},'.');
                cstr = strsplit(cstr{1},'_');
    
                % Read in current simulated image
                ImgSim = double(imread(fullfile(dirSims, simNamelist{i})));
                % Find Laplacian of current simulated image
                lapImgSim = conv2(ImgSim, lKernel, 'same');  
    
                % Collect ratio with Laplacian
                cc_l = [];
                % Identifier for patch
                index = 1;
                
                % Create patches in real image (ImgR_patch),
                % and simulated image(ImgSim_patch) 
                % For patches: row,col start at intv,intv
                for prow = intv:szPatch+intv:img_h-szPatch
                    for pcol = intv:szPatch+intv:img_w-szPatch
                        % Patch of simulated images 
                        lapImgSim_patchRef = lapImgSimRef(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
                        lapImgSim_patch = lapImgSim(prow:prow+szPatch-1,pcol:pcol+szPatch-1);
    
                        m = turbulence_metric_noBL(lapImgSim_patchRef, lapImgSim_patch);
                        cc_l(index) = m; % Save results of all patches
                        index = index + 1;
                   end
                end
                % Calculate mean metric of all patches for this image and save to
                % table TmL
                avg_rl = mean(cc_l); % Laplacian
                std_rl = std(cc_l);
                TmL(indT,:) = {rng zm string(cstr{1}) simNamelist{i}  numPatches*numPatches avg_rl std_rl}; % Laplacian
                indT = indT + 1;
            end
        end
    end
end    

varnames = {'range', 'zoom', 'cn2str', 'Simfilename','numPatches', 'simMetric', 'stdPatches'};
TmL = renamevars(TmL, TmL.Properties.VariableNames, varnames);
TmL.filename = string(TmL.Simfilename);

% Use "turbNums.csv" (created by Python file) to find r0 for plotting
Tr0 = readtable(data_root + "modifiedBaselines\SimImgs_VaryingCn2\turbNums.csv");
Tr0.strcn2 = string(Tr0.cn2);
Tr0.strcn2 = strrep(Tr0.strcn2,'-','');

for k = 1:height(TmL)
    indK = find(Tr0.range == TmL.range(k) & Tr0.strcn2 == TmL.cn2str(k));
    TmL.r0(k) = Tr0.r0(indK);
    TmL.cn2(k) = Tr0.cn2(indK);
end

% Save TmL table
writetable(TmL, data_root + "modifiedBaselines\SimImgs_SameCn2\TmL_650Plus.csv");

%% Plots
for rngP = rangeV

    for cnP = cn2str
        figure()
        x = 1:20;

        % Get r0 for real image in fileA - to use in plots
        %fileA = data_root + "combined_sharpest_images_withAtmos.xlsx";
        %T_atmos = readtable(fileA);
        legendL = [];        
        for zmP = zoom
            % Get TmL data to plot
            ccnP = split(cnP, 'c');
            ccnP = ccnP(2);
            indP = find(TmL.range == rngP & TmL.zoom == zmP & TmL.cn2str == ccnP);
            r0V = TmL.r0(indP);
            r0V = num2str(r0V(1)*100);
            cnV = TmL.cn2(indP);
            cnV = num2str(cnV(1));
            plot(x,TmL.simMetric(indP),'-o',...
                        'LineWidth',1,...
                        'MarkerSize',2)
            xlim([1,20])
            ylim([0.5,1.0])
            xlabel("Simulated Image Number")
            ylabel("Metric")
            % Setup legend entry
            txt = "Z" + num2str(zmP);
            legendL = [legendL; txt];
            hold on;
        end
        grid on
        title("Turbulence Metric: Range " + num2str(rngP) + ", r0 " + r0V +", Cn2 " + cnV + " " + titlePtch) 
        x0=10;
        y0=10;
        width=600;
        ht=400;
        legend(legendL, 'location','southeast')
        set(gcf,'position',[x0,y0,width,ht]) 
        f = gcf;
        dirOutF = dirOut + "\r" + num2str(rngP);
        fileN = fullfile(dirOutF, "SimOnly_Lr"+ num2str(rngP) + "_c" + cnP + strPtch + ".png");
        exportgraphics(f,fileN,'Resolution',300)
        close all
    end
end

