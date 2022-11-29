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


filename = "D:/data/turbulence/sharpest/z5000/baseline_z5000_r1000.png";

img = double(imread(filename));

[img_h, img_w, img_c] = size(img);

if(img_c == 3)
   img = img(:, :, 2); 
end

img_blur = zeros(img_h * img_w, 1);

% create the correct matlab pointers to pass into the function
img_t = libpointer('doublePtr', img);
img_blur_t = libpointer('doublePtr', img_blur);

D = 0.095;
L = 1000;
Cn2 = 1e-13;
wavelength = 525e-9;
obj_size = img_w * 0.004276;

calllib(lib_name, 'init_turbulence_params', img_w, D, L, Cn2, wavelength, obj_size);

for idx=1:20
    calllib(lib_name, 'apply_turbulence', img_w, img_h, img_t, img_blur_t);

    img_blur = reshape(img_blur_t.Value, [img_h, img_w])';

    figure(1);
    image(uint8(img_blur));
    colormap(gray(256))
    pause(1);

end

bp = 1;

unloadlibrary(lib_name);
