% This file was created from tub_sim_lib_example.m to create simulated images of each real
% image, using the range and Cn2 of real image and the modified baseline
% images.  Note that the modified baseline images vary by range and zoom
% value.
% This script creates 20 simulated images for each real image.

format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[startpath,  filename, ext] = fileparts(full_path);
plot_num = 1;

cd(startpath);

clear tempdir
setenv('TMP',startpath);
commandwindow;

%% load the dll/so file

lib_path = '..\turb_sim_lib\build\Release\';
lib_name = 'turb_sim';
hfile = '../common/include/turb_sim_lib.h';

if(~libisloaded(lib_name))
    [notfound, warnings] = loadlibrary(strcat(lib_path,lib_name,'.dll'), hfile);
end

if(~libisloaded(lib_name))
   fprintf('\nThe %s library did not load correctly!',  strcat(lib_name,'.dll'));    
end

% show all the available functions and 
% libfunctionsview(lib_name);
% pause(1);

% Setup data directories
platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

% Location of modified baseline images
dirBL = data_root + "modifiedBaselines\";
% combined_sharpest_images_withAtmos.xlsx in data_root directory

% Steps:
% 1.  Get list of all modified BL images.
% 2.  Determine range/zoom from filename in list of modified BL images.
% 3.  Filter list of images for range/zoom to only process images of user
%     defined range/zoom.
% 4.  Use range and Cn2 value to create simulated images
% 5.  Create 20 simulated images

%rangeV = [700];
rangeV = 600:50:1000;
%zoom = [2500];
zoom = [2000, 2500, 3000, 3500, 4000, 5000];

% Common parameters
% Images will be created with the following Cn2 values
cn2Values = [7e-16, 8e-16, 9e-16, ...
            1e-15, 2e-15, 3e-15, 4e-15, 5e-15, 6e-15, 7e-15, 8e-15, 9e-15,...
            1e-14, 2e-14, 3e-14, 4e-14, 5e-14, 6e-14, 7e-14, 8e-14, 9e-14,...
            1e-13, 2e-13, 3e-13, 4e-13, 5e-13, 6e-13, 7e-13, 8e-13, 9e-13,...
            1e-12];
numSims = 20;  % Number of simulated images created of same zoom/range/cn2 
D = 0.095;
wavelength = 525e-9;

% Get object size from real image data in fileA 
fileA = data_root + "combined_sharpest_images_withAtmos.xlsx";
T_atmos = readtable(fileA);
varnamesA = {'Date', 'Time', 'Time_secs', 'range', 'zoom', 'focus', 'img_filename', ...
    'img_height', 'img_width', 'pixel_step', 'start', 'stop', 'obj_size', 'Temperature', ...
    'Humidity', 'Wind_speed', 'Wind_dir', 'Bar_pressure', 'Solar_load', 'Cn2', 'r0' };
T_atmos = renamevars(T_atmos, T_atmos.Properties.VariableNames, varnamesA);

% Get list of all mod BL images
blFiles = dir(fullfile(dirBL, '*.png'));
blNames = {blFiles(~[blFiles.isdir]).name}; % All filenames in Modified BL dir

for rng = rangeV
    % In blNames, find the needed baseline image file for zm/rng
    for zm = zoom  
        display("Range " + num2str(rng) + " Zoom " + num2str(zm))
        % Filter by range and zoom to get file names of range/zoom
        if rng < 1000
            patt = "_z" + num2str(zm) + "_r0" + num2str(rng);
        else
            patt = "_z" + num2str(zm) + "_r" + num2str(rng);
        end 
        % Find modified baseline filename
        for i = 1:length(blNames)
            if contains(blNames{:,i},patt)
                blfilename = blNames{:,i};
                %display(namelist{ind})
                break;
            end
        end
        % Using this mod baseline file, create images for each Cn2 value
        % Get obj_size from combined_sharpest_images_withAtmos.xlsx 
        indA = find(T_atmos.range == rng & T_atmos.zoom == zm);
        %Cn2 = T_atmos.Cn2(indA); 
        obj_size = T_atmos.obj_size(indA);

        % Read in baseline image
        readFile = fullfile(dirBL, blfilename);
        img = double(imread(readFile));

        [img_h, img_w, img_c] = size(img);

        if(img_c == 3)
            img = img(:, :, 2); 
        end

        for Cn2 = cn2Values
            img_blur = zeros(img_h * img_w, 1);
            % create the correct matlab pointers to pass into the function
            img_t = libpointer('doublePtr', img);
            img_blur_t = libpointer('doublePtr', img_blur);
            
            calllib(lib_name, 'init_turbulence_params', img_w, D, rng, Cn2, wavelength, obj_size);
       
            % Create 20 simulated images (make sure to save as uint8)
            for idx=1:20
                calllib(lib_name, 'apply_turbulence', img_w, img_h, img_t, img_blur_t);
            
                img_blur = reshape(img_blur_t.Value, [img_h, img_w])';
                img_blur = uint8(img_blur);
                cn2s = strrep(num2str(Cn2), '-','');
                fileI = "NewSim_r" + num2str(rng) + "_z" + num2str(zm) + "_c" + cn2s +     "_N" + num2str(idx) + ".png";
                pathI = data_root + "modifiedBaselines\NewSimulations\ByVaryingCn2\" + fileI;
                imwrite(img_blur, pathI);
            
            end
        end

    end
end

unloadlibrary(lib_name);
