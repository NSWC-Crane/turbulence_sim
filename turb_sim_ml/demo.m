format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[scriptpath,  filename, ext] = fileparts(full_path);

%%

img_in  = im2double(imread(strcat(scriptpath, '/../data/checker_board_32x32.png')));
if size(img_in,3)~=1
    img  = rgb2gray(img_in);
end

% crop instead of resizing
img = img(1:256, 1:256);
%img = imresize(img,[256,256]);

params.t_params.D = 0.095;
params.t_params.L = 1000;
params.t_params.d = 1.38;
params.t_params.Cn2 = 1e-14;
params.t_params.lambda = 0.525e-6;

opts.frames = 1;

temp = sim_fun(img, params, opts);

out_stack = uint8(256*temp);
