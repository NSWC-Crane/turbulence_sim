% Return standard deviation of simulated images
% by Range and Zoom
% Use new simulated images
% x-axis is zoom
% Collect table by range with columns for zoom, mean, std, size
% Plot results

clear
clc

rangeV = 600:50:1000;
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
    %% Set up a table to collect results by range value
    % Create tables by range and save by range
    % Entries: range, zoom, filename, image height, image width, mean pix
    % value, std pix value
    numFiles = 20000;
    col_label = ["ImgPath","Filename","Range","Zoom","ImgHt","ImgWd","MeanPixVal","StdPixVal"];
    vartypes = {'string','string','uint16','uint16','uint16','uint16','double','double'};
    TSim = table('Size', [numFiles, length(col_label)], 'VariableTypes', vartypes);
    TSim.Properties.VariableNames = col_label.';
    indT = 1;

    for zoom = zoomV
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
    
            % Read in image
            img = double(imread(fullfile(dirSims, simNamelist{i})));
        
            % Find size, mean, std
            [img_h, img_w] = size(img);
            mean_img = mean(img,"all");
            std_img = std(img,0,'all');
    
            % Enter results above into table with zoom and range
            TSim(indT,["ImgPath","Filename","Range","Zoom","ImgHt","ImgWd","MeanPixVal","StdPixVal"]) = ...
                        {dirSims, simNamelist{i}, rng, zoom, img_h, img_w, mean_img, std_img };
            indT = indT + 1;
        end
        
    end

    % Remove unused rows in table
    TSim = TSim(TSim.Range > 0,:);
    writetable(TSim, data_root + "Results_StdImages\tSim_" + num2str(rng) + ".csv");

    % Plot by range
    f = figure();
    p = scatter(TSim,"Zoom", "MeanPixVal");
    p.Marker ="x";
    hold on
    q = scatter(TSim,"Zoom", "StdPixVal");
    q.Marker = "*";
    q.MarkerFaceColor = 'm';
    xlabel("Zoom")
    ylabel("Pixel Value Mean/Standard Deviation")
    title("Range " + num2str(rng) + ": Simulated Images Varying Cn2(Mean & STD)")
    yticks(0:10:255)
    ylim([0,255])
    grid on
    set(f,'position',([100,100,400,700]),'color','w')
    fileOut = data_root + "Results_StdImages\tSim_" + num2str(rng) + ".png";
    exportgraphics(f,fileOut,'Resolution',300)

end



