clear all; close all; clc

img_in  = im2double(imread('./data/Pattern1_GT.png'));
if size(img_in,3)~=1
    img_in  = rgb2gray(img_in);
end
img = imresize(img_in,[256,256]);

params.t_params.L = 10;
temp = sim_fun(img,params);

out_stack = uint8(256*temp);