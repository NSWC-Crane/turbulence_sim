format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[startpath,  baseline_filename, ext] = fileparts(full_path);
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

%% Setup data directories
platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

r = 600;
z = 2000;

%% load in the sharpest images
sharpest_dir = data_root + "sharpest\z" + num2str(z) + "\" + num2str(r, '%04d') + "\";

listing = dir(strcat(sharpest_dir,'*.png'));

sharpest_img = cell(length(listing),1);

for idx=1:length(listing)
    img = double(imread(strcat(listing(idx).folder, filesep, listing(idx).name)));

    [~, ~, img_c] = size(img);

    if(img_c == 3)
       img = img(:, :, 2); 
    end
    
    sharpest_img{idx} = img;

end

%% load the baseline image
% baseline_filename = data_root + "ModifiedBaselines\Mod_baseline_z" + num2str(z) + "_r" + num2str(r, '%04d') + ".png";
baseline_filename = "../data/random_image_512x512.png";
img_ref = double(imread(baseline_filename));

[img_h, img_w, img_c] = size(img_ref);

% if(img_c == 3)
%    img_ref = img_ref(:, :, 2); 
% end

%% setup image turbulence
img_blur = zeros(img_h * img_w * 3 , 1);

% create the correct matlab pointers to pass into the function
img_t = libpointer('doublePtr', img_ref);
img_blur_t = libpointer('doublePtr', img_blur);

use_color = int8(1);
zoom = 2000;
D = 0.095;
L = 600;
Cn2 = 1e-15;
wavelength = 525e-9;
obj_size = img_w * get_pixel_size(L, zoom);

calllib(lib_name, 'init_turbulence_generator', img_w, D, L, Cn2, obj_size, use_color);

%% run several images
tb_metric = [];
tb_metric2 = [];

for idx=1:50
    calllib(lib_name, 'apply_rgb_turbulence', img_w, img_h, img_t, img_blur_t);

%     img_blur = reshape(img_blur_t.Value, [img_h, img_w])';
    img_blur = cat(3, reshape(img_blur_t.Value(3:3:end), [img_h, img_w])', reshape(img_blur_t.Value(2:3:end), [img_h, img_w])', reshape(img_blur_t.Value(1:3:end), [img_h, img_w])');
%     img_blur = cat(3, reshape(img_blur_t.Value(:,1), [img_h, img_w])', reshape(img_blur_t.Value(:,2), [img_h, img_w])', reshape(img_blur_t.Value(:,3), [img_h, img_w])');


%     tb_metric(idx) = turbulence_metric_noBL(sharpest_img{1}, img_blur);
%     tb_metric2(idx) = tb_metric_v2(sharpest_img{1}, img_blur);
        
%     montage = cat(2, sharpest_img{1}, 255*ones(img_h, 10), img_blur);
    figure(1);
    image(uint8(img_blur));
%     colormap(gray(256))
    pause(0.001);

end

bp = 1;

%%
fprintf('unloading library...\n');
unloadlibrary(lib_name);

fprintf('complete!\n');

