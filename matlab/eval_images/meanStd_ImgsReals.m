% Return standard deviation of real(sharpest) images
% by Range and Zoom
% Use all reals
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


%% Go through range
for rng = rangeV

    % Create tables by range and save by range
    % Entries: range, zoom, filename, image height, image width, mean pix
    % value, std pix value
    %% Set up a table to collect results by range value
    numFiles = 20000;
    col_label = ["ImgPath","Filename","Range","Zoom","ImgHt","ImgWd","MeanPixVal","StdPixVal"];
    vartypes = {'string','string','uint16','uint16','uint16','uint16','double','double'};
    TReal = table('Size', [numFiles, length(col_label)], 'VariableTypes', vartypes);
    TReal.Properties.VariableNames = col_label.';
    indT = 1;
    for zoom = zoomV
        % Get real image file names
        [dirSharp, ImgNames] = GetRealImageFilenames(data_root, rng, zoom);
        for imgN = ImgNames
    
            % Read in image
            img = double(imread(fullfile(dirSharp, imgN)));
            img = img(:,:,2);  % Only green channel
        
            % Find size, mean, std
            [img_h, img_w] = size(img);
            mean_img = mean(img,"all");
            std_img = std(img,0,'all');
    
            % Enter results above into table with zoom and range
            TReal(indT,["ImgPath","Filename","Range","Zoom","ImgHt","ImgWd","MeanPixVal","StdPixVal"]) = ...
                        {dirSharp, imgN, rng, zoom, img_h, img_w, mean_img, std_img };
            indT = indT + 1;
        end
        
    end

    % Remove unused rows in table
    TReal = TReal(TReal.Range > 0,:);
    writetable(TReal, data_root + "Results_StdImages\tReal_" + num2str(rng) + ".csv");

    % Plot by range
    f = figure();
    p = scatter(TReal,"Zoom", "MeanPixVal");
    p.Marker ="x";
    hold on
    q = scatter(TReal,"Zoom", "StdPixVal");
    q.Marker = "*";
    q.MarkerFaceColor = 'm';
    xlabel("Zoom")
    ylabel("Pixel Value Mean/Standard Deviation")
    title("Range " + num2str(rng) + ": Real Sharpest Images(Mean & STD)")
    yticks(0:10:255)
    ylim([0,255])
    grid on
    set(f,'position',([100,100,400,700]),'color','w')
    fileOut = data_root + "Results_StdImages\tReal_" + num2str(rng) + ".png";
    exportgraphics(f,fileOut,'Resolution',300)

end



