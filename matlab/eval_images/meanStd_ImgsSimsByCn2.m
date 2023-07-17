% Return standard deviation of simulated images
% by Range and Zoom
% Use new simulated images
% x-axis is zoom
% Collect table by range with columns for zoom, mean, std, size
% Plot results

clear
clc

rangeV = 600:50:650;
zoomV = [2000,2500,3000,3500,4000,5000];

platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

dirSims = data_root + "modifiedBaselines\NewSimulations\ByVaryingCn2\";

%% Get simulated image file names
simFiles = dir(fullfile(dirSims, '*.png'));
SimImgNames = {simFiles(~[simFiles.isdir]).name};

%% Go through range
for rng = rangeV
    for zoom = zoomV
        %% Set up a table to collect results by range value
        % Create tables by range and save by range
        % Entries: range, zoom, filename, image height, image width, mean pix
        % value, std pix value
        numFiles = 20000;
        col_label = ["ImgPath","Filename","Range","Zoom","Cn2","ImgHt","ImgWd","MeanPixVal","StdPixVal"];
        vartypes = {'string','string','uint16','uint16','double','uint16','uint16','double','double'};
        TSim = table('Size', [numFiles, length(col_label)], 'VariableTypes', vartypes);
        TSim.Properties.VariableNames = col_label.';
        indT = 1;

        % Get simulated image file names
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

        for i = 1:length(simNamelist)
            % Get image Cn2 value from filename
             % cstr:  cn2 in filename (used to get r0 later)
            cstr = strsplit(simNamelist{i},'_c');
            cstr = strsplit(cstr{2},'.');
            cstr = strsplit(cstr{1},'_');
            cstr = cstr{1};
            cstr = insertAfter(cstr, 'e','-');
            cn2 = str2double(cstr);
            % Read in image
            img = double(imread(fullfile(dirSims, simNamelist{i})));
        
            % Find size, mean, std
            [img_h, img_w] = size(img);
            mean_img = mean(img,"all");
            std_img = std(img,0,'all');
    
            % Enter results above into table with zoom and range
            TSim(indT,["ImgPath","Filename","Range","Zoom","Cn2","ImgHt","ImgWd","MeanPixVal","StdPixVal"]) = ...
                        {dirSims, simNamelist{i}, rng, zoom, cn2, img_h, img_w, mean_img, std_img };
            indT = indT + 1;
        end

        % Remove unused rows in table
        TSim = TSim(TSim.Range > 0,:);
        writetable(TSim, data_root + "Results_StdImages\tSimCn2_R" + num2str(rng) + "_Z" + num2str(zoom)+ ".csv");

        % Plot by range and zoom
        f = figure();
        p = scatter(TSim,"Cn2", "MeanPixVal");
        p.Marker ="x";
        hold on
        q = scatter(TSim,"Cn2", "StdPixVal");
        q.Marker = "*";
        q.MarkerFaceColor = 'm';
        xlabel("Cn2")
        ylabel("Pixel Value Mean/Standard Deviation")
        title("Range " + num2str(rng) + " Zoom " + num2str(zoom) + " Simulated Images Varying Cn2(Mean & STD)")
        yticks(0:10:255)
        ylim([0,255])
        grid on
        set(gca,'xscale','log')
        set(f,'position',([100,100,500,700]),'color','w')
        fileOut = data_root + "Results_StdImages\tSimCn2_R" + num2str(rng) + "_Z" + num2str(zoom)+ ".png";
        exportgraphics(f,fileOut,'Resolution',300)
        
    end 

end



